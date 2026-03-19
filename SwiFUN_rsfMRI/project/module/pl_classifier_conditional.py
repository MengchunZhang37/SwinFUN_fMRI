import hashlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn

from .pl_classifier import LitClassifier


class LitConditionalClassifier(LitClassifier):
    def __init__(self, data_module, **kwargs):
        super().__init__(data_module, **kwargs)

        self.cond_enabled = bool(getattr(self.hparams, "rfmri_conditional", False))
        self.cond_sources = list(getattr(self.hparams, "rfmri_condition_sources", ["subject_name", "sex"]))
        self.cond_dim = int(getattr(self.hparams, "rfmri_condition_dim", 64))
        self.cond_subject_vocab_size = int(getattr(self.hparams, "rfmri_subject_vocab_size", 8192))

        self._numeric_condition_keys = [k for k in self.cond_sources if k in {"sex", "target", "tr"}]
        self._use_subject_condition = "subject_name" in self.cond_sources

        if self.cond_enabled and getattr(self.hparams, "downstream_task", "") == "rfMRI_next":
            feature_dim = int(getattr(self.model, "num_features", self.hparams.embed_dim * (self.hparams.c_multiplier ** (len(self.hparams.depths) - 1))))

            if self._use_subject_condition:
                self.subject_condition_embedding = nn.Embedding(self.cond_subject_vocab_size, self.cond_dim)
            else:
                self.subject_condition_embedding = None

            if len(self._numeric_condition_keys) > 0:
                self.numeric_condition_proj = nn.Sequential(
                    nn.Linear(len(self._numeric_condition_keys), self.cond_dim),
                    nn.GELU(),
                    nn.Linear(self.cond_dim, self.cond_dim),
                )
            else:
                self.numeric_condition_proj = None

            self.condition_fuser = nn.Sequential(
                nn.LayerNorm(self.cond_dim),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.GELU(),
            )
            self.condition_to_gamma = nn.Linear(self.cond_dim, feature_dim)
            self.condition_to_beta = nn.Linear(self.cond_dim, feature_dim)

            nn.init.zeros_(self.condition_to_gamma.weight)
            nn.init.zeros_(self.condition_to_gamma.bias)
            nn.init.zeros_(self.condition_to_beta.weight)
            nn.init.zeros_(self.condition_to_beta.bias)
        else:
            self.subject_condition_embedding = None
            self.numeric_condition_proj = None
            self.condition_fuser = None
            self.condition_to_gamma = None
            self.condition_to_beta = None

    def _subject_names_to_indices(self, subjects, device):
        indices = []
        for subject in subjects:
            if isinstance(subject, torch.Tensor):
                subject = subject.item()
            digest = hashlib.md5(str(subject).encode("utf-8")).hexdigest()
            indices.append(int(digest, 16) % self.cond_subject_vocab_size)
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _extract_numeric_condition(self, target_value, tr, sex, batch_size, device, dtype):
        values = []
        for key in self._numeric_condition_keys:
            if key == "sex":
                tensor = torch.as_tensor(sex, device=device)
                tensor = tensor.reshape(batch_size, -1).to(dtype)
                values.append(tensor[:, :1])
            elif key == "tr":
                tensor = torch.as_tensor(tr, device=device)
                tensor = tensor.reshape(batch_size, -1).to(dtype)
                values.append(tensor[:, :1])
            elif key == "target":
                if torch.is_tensor(target_value):
                    tensor = target_value.to(device=device, dtype=dtype).reshape(batch_size, -1)
                    values.append(tensor.mean(dim=1, keepdim=True))
        if not values:
            return None
        return torch.cat(values, dim=1)

    def _build_condition_vector(self, subj, target_value, tr, sex, feat):
        if not self.cond_enabled or self.condition_fuser is None:
            return None

        batch_size = feat.shape[0]
        dtype = feat.dtype
        device = feat.device
        parts = []

        if self._use_subject_condition and self.subject_condition_embedding is not None:
            subject_ids = self._subject_names_to_indices(subj, device=device)
            parts.append(self.subject_condition_embedding(subject_ids).to(dtype))

        numeric = self._extract_numeric_condition(target_value, tr, sex, batch_size, device, dtype)
        if numeric is not None and self.numeric_condition_proj is not None:
            parts.append(self.numeric_condition_proj(numeric))

        if not parts:
            return None
        fused = torch.stack(parts, dim=0).sum(dim=0)
        return self.condition_fuser(fused)

    def _apply_condition_to_feature(self, feat, cond_vec):
        if cond_vec is None or self.condition_to_gamma is None or self.condition_to_beta is None:
            return feat

        view_shape = [feat.shape[0], feat.shape[1]] + [1] * (feat.ndim - 2)
        gamma = self.condition_to_gamma(cond_vec).view(*view_shape)
        beta = self.condition_to_beta(cond_vec).view(*view_shape)
        return feat * (1.0 + gamma) + beta

    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()

        if getattr(self.hparams, "downstream_task", "") != "rfMRI_next":
            return super()._compute_logits(batch, augment_during_training=augment_during_training)

        if augment_during_training:
            fmri = self.augment(fmri)

        objective = self._get_rfmri_objective()
        baseline_only = self._get_rfmri_baseline_only()
        B, C, H, W, D, T_full = fmri.shape

        if objective == "masked_reconstruction":
            if baseline_only != "none":
                if baseline_only not in {"zero", "mean"}:
                    raise ValueError(
                        f"rfmri_baseline_only={baseline_only} is incompatible with masked_reconstruction"
                    )
                target = fmri.clone()
                logits_main = self._rfmri_baseline_full_window(target, baseline_only)
                self._last_rfMRI_logits_res = None
                self._last_rfMRI_loss_mask = None
                return subj, logits_main, target

            masked_fmri, recon_loss_mask, keep_mask = self._apply_rfmri_recon_mask(fmri)
            feat = self.model(masked_fmri)
            cond_vec = self._build_condition_vector(subj, target_value, tr, sex, feat)
            feat = self._apply_condition_to_feature(feat, cond_vec)
            logits_main = self._project_to_voxels_time(self.output_head(feat), out_T=T_full)
            target = fmri.clone()

            if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                spatial_mask = self.MNI152_mask[None, None, ...].to(logits_main.dtype).unsqueeze(-1)
                logits_main = logits_main * spatial_mask
                target = target * spatial_mask

            if self.residual_head is not None:
                base_recon = self._rfmri_recon_baseline(
                    masked_fmri,
                    keep_mask,
                    kind=str(getattr(self.hparams, "rfmri_recon_base", "zero"))
                )
                res_raw = self._project_to_voxels_time(self.residual_head(feat), out_T=T_full)
                logits_res = base_recon + res_raw
                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    spatial_mask = self.MNI152_mask[None, None, ...].to(logits_res.dtype).unsqueeze(-1)
                    logits_res = logits_res * spatial_mask
                self._last_rfMRI_logits_res = logits_res
            else:
                self._last_rfMRI_logits_res = None
            self._last_rfMRI_loss_mask = recon_loss_mask
            return subj, logits_main, target

        if self.hparams.pred_context is None or self.hparams.pred_horizon is None:
            T_ctx = T_full // 2
            T_hz = T_full - T_ctx
        else:
            T_ctx = int(self.hparams.pred_context)
            T_hz = int(self.hparams.pred_horizon)
        assert T_ctx + T_hz <= T_full, "pred_context + pred_horizon > T_full"

        ctx = fmri[..., :T_ctx]
        y_future = fmri[..., T_ctx:T_ctx + T_hz]

        if baseline_only != "none":
            if baseline_only not in {"persist", "ctxmean"}:
                raise ValueError(
                    f"rfmri_baseline_only={baseline_only} is incompatible with next_token"
                )
            preds_main = []
            ctx_roll = ctx
            for _ in range(T_hz):
                base_raw = self._rfmri_baseline_one_step(ctx_roll, baseline_only, apply_mask=False)
                preds_main.append(base_raw)
                ctx_roll = torch.cat([ctx_roll, base_raw], dim=-1)
                if getattr(self.hparams, "ctx_update", "sliding") == "sliding":
                    if ctx_roll.shape[-1] > T_ctx:
                        ctx_roll = ctx_roll[..., -T_ctx:]
                elif self.hparams.ctx_update == "growing":
                    pass
                else:
                    raise ValueError(f"Unknown ctx_update={self.hparams.ctx_update}")

            logits_main = torch.cat(preds_main, dim=-1)
            target = y_future.clone()
            self._last_rfMRI_logits_res = None
            self._last_rfMRI_loss_mask = None
            return subj, logits_main, target

        tf_ratio = self._tf_ratio()
        preds_main = []
        preds_res = [] if (self.residual_head is not None) else None
        for t in range(T_hz):
            feat_t = self.model(ctx)
            cond_vec = self._build_condition_vector(subj, target_value, tr, sex, feat_t)
            feat_t = self._apply_condition_to_feature(feat_t, cond_vec)

            pred_t_raw = self._project_to_voxels_time(self.output_head(feat_t), out_T=1)
            preds_main.append(pred_t_raw)

            pred_res_raw = None
            if self.residual_head is not None:
                res_raw = self._project_to_voxels_time(self.residual_head(feat_t), out_T=1)
                base_raw = self._rfmri_baseline_one_step(ctx, getattr(self.hparams, "residual_base", "persist"), apply_mask=False)
                pred_res_raw = base_raw + res_raw

                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    m = self.MNI152_mask[None, None, ...].to(pred_res_raw.dtype).unsqueeze(-1)
                    preds_res.append(pred_res_raw * m)
                else:
                    preds_res.append(pred_res_raw)

            use_teacher = (torch.rand((), device=ctx.device) < tf_ratio).item()
            if use_teacher:
                next_frame = y_future[..., t:t + 1]
            else:
                next_frame = pred_res_raw if pred_res_raw is not None else pred_t_raw

            ctx = torch.cat([ctx, next_frame], dim=-1)
            if getattr(self.hparams, "ctx_update", "sliding") == "sliding":
                if ctx.shape[-1] > T_ctx:
                    ctx = ctx[..., -T_ctx:]
            elif self.hparams.ctx_update == "growing":
                pass
            else:
                raise ValueError(f"Unknown ctx_update={self.hparams.ctx_update}")

        logits_main = torch.cat(preds_main, dim=-1)
        target = y_future.clone()
        self._last_rfMRI_logits_res = torch.cat(preds_res, dim=-1) if preds_res is not None else None
        self._last_rfMRI_loss_mask = None
        return subj, logits_main, target

    def _extract_sample_embeddings(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()

        if getattr(self.hparams, "downstream_task", "") != "rfMRI_next":
            return super()._extract_sample_embeddings(batch, augment_during_training=augment_during_training)

        if augment_during_training:
            fmri = self.augment(fmri)

        objective = self._get_rfmri_objective()
        if objective == "next_token":
            T_ctx, _ = self._get_rfmri_ctx_horizon(fmri)
            enc_in = fmri[..., :T_ctx]
        else:
            enc_in = fmri

        feat = self.model(enc_in)
        cond_vec = self._build_condition_vector(subj, target_value, tr, sex, feat)
        feat = self._apply_condition_to_feature(feat, cond_vec)
        return subj, self._pool_backbone_feature(feat)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LitClassifier.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("Conditional rfMRI")
        group.add_argument("--rfmri_conditional", action="store_true",
                           help="Enable conditional modulation for rfMRI_next")
        group.set_defaults(rfmri_conditional=False)
        group.add_argument("--rfmri_condition_sources", nargs="+", default=["subject_name", "sex"],
                           choices=["subject_name", "sex", "target", "tr"],
                           help="Metadata sources used to build the rfMRI condition vector")
        group.add_argument("--rfmri_condition_dim", type=int, default=64,
                           help="Hidden size of the rfMRI condition embedding")
        group.add_argument("--rfmri_subject_vocab_size", type=int, default=8192,
                           help="Hash-bucket size for subject-id embedding when subject_name is used as condition")
        return parser
