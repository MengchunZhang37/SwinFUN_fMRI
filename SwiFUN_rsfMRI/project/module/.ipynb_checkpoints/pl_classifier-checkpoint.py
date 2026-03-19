import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy

import torchmetrics
import torchmetrics.classification
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from torchmetrics import  PearsonCorrCoef # Accuracy,
from torchmetrics.regression import R2Score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb


from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

from einops import rearrange

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer
from datetime import datetime
import gzip, pickle
import copy


def contrast_mse_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          subjects) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据论文 2.3 节定义 RC loss 的两个组成：
          - within_subj_loss = L_R = 平均重建 MSE（逐样本对齐的 pred_i vs target_i）
          - across_subj_loss = L_C = 仅在“跨 subject”对上，计算 pred_j vs target_i 的两两 MSE 并求均值
                                (排除 i==j 以及 subj_i == subj_j 的所有对)
        参数
          pred:   [B, F]  已按需掩膜/展平后的模型输出
          target: [B, F]  已按需掩膜/展平后的真值
          subjects: 长度为 B 的 subject 标识（list/ndarray/tensor，元素可为 str 或 int）
        返回
          (within_subj_loss, across_subj_loss)  都是标量 tensor
        """
        assert pred.ndim == 2 and target.ndim == 2 and pred.shape == target.shape
        B, Fdim = pred.shape
        device = pred.device

        # 1) L_R: 对齐样本的一对一重建 MSE（论文式 (54~61)）
        within = F.mse_loss(pred, target, reduction="mean")

        # 2) L_C: 跨 subject 的两两 MSE 平均（论文式 (62~66)）
        # 计算 pred_j vs target_i 的 pairwise MSE（用范数恒等式避免构造 (B,B,F) 大张量）：
        # M[j,i] = (||pred_j||^2 + ||target_i||^2 - 2*pred_j·target_i) / F
        # pairwise mse: pred[j] vs target[i]
        p2 = (pred * pred).sum(dim=1, keepdim=True)          # [B,1]
        t2 = (target * target).sum(dim=1, keepdim=True).t()  # [1,B]
        cross = pred @ target.t()                             # [B,B]
        pair_mse = (p2 + t2 - 2.0 * cross) / float(Fdim)      # [B,B] row=j col=i

        # subjects -> tensor codes
        if isinstance(subjects, torch.Tensor):
            s = subjects.detach().cpu()
        else:
            s = torch.tensor(list(subjects), dtype=torch.long) if isinstance(list(subjects)[0], (int, np.integer)) \
                else None
    
        if s is None:
            # string case: map to ints
            uniq = {v:i for i,v in enumerate(list(subjects))}
            s = torch.tensor([uniq[v] for v in subjects], dtype=torch.long)
    
        s = s.to(device)
        same_subj = (s.view(B,1) == s.view(1,B))              # [B,B] row=i col=j (sample index)
        diag = torch.eye(B, dtype=torch.bool, device=device)
    
        # pair_mse is [row=j, col=i] so we need mask in same orientation:
        # invalid[j,i] if i==j OR subj_i==subj_j
        invalid = diag | same_subj
        invalid = invalid.t()  # now [row=j, col=i]
    
        valid = ~invalid
        across = pair_mse[valid].mean() if valid.any() else torch.zeros((), device=device)
    
        return within, across

class LitClassifier(pl.LightningModule):
    def __init__(self, data_module, **kwargs):
        super().__init__()
        

        while True:
            try:
                self.save_hyperparameters(kwargs)
                break
            except TypeError as e:
                import re
                match = re.findall(r"unexpected keyword argument '(\w+)'", str(e))
                if match:
                    key = match[0]
                    print(f"[WARNING] Removing illegal hparam key: {key}")
                    kwargs.pop(key)
                else:
                    raise e
        
        run_id = kwargs.get("id", None)
        if not run_id or run_id == "None":
            run_id = f"{self.hparams.dataset_name}_{self.hparams.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run_id = run_id
       
        # you should define target_values at the Dataset classes
        if (self.hparams.downstream_task in ['tfMRI_3D', 'rfMRI_next']):
            scaler = None
        elif self.hparams.label_scaling_method == 'standardization':
            target_values = data_module.train_dataset.target_values
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax': 
            target_values = data_module.train_dataset.target_values
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler

        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)
        self.metric = Metrics()
        
        # === only-save best/last ===
        if getattr(self.hparams, "downstream_task_type", "default") == "classification":
            self._monitor_key = "valid_acc"
            self._monitor_mode = "max"  # acc larger better
        else:
            # tfMRI_3D / rfMRI_next / regression - MSE
            self._monitor_key = "valid_mse"
            self._monitor_mode = "min"   # mse lower better
        self._best_val = None


        # Heads
        if self.hparams.downstream_task in ['tfMRI_3D', 'rfMRI_next']: # 3D volume prediction
            self.output_head = nn.Identity() #load_model("swinunetr", self.hparams)
        elif self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            self.output_head = load_model("clf_mlp", self.hparams)
            #self.clf = load_model("clf_mlp", self.hparams)
        elif self.hparams.downstream_task_type == 'regression':
            self.output_head = load_model("reg_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")

        #add residual branch
        if getattr(self.hparams, "use_residual_branch", False) and getattr(self.hparams, "downstream_task", "") == "rfMRI_next":
            self.residual_head = copy.deepcopy(self.output_head)
        else:
            self.residual_head = None
        
        
        if getattr(self.hparams, "mask_input", False):
            mask_path = os.path.join(self.hparams.image_path, "ROI", self.hparams.mask_filename)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            # Load mask, must be a tensor saved by torch.save
            m = torch.load(mask_path, map_location="cpu")
            if not isinstance(m, torch.Tensor):
                raise ValueError("Mask file must be a torch.Tensor saved by torch.save")

            # Squeeze to ensure shape [96,96,96]
            if m.ndim == 4 and m.shape[0] == 1:
                m = m[0]

            # Dataset-specific padding (if needed)
            background_value = m.flatten()[0].item() if m.numel() > 0 else 0
            if "UKB" in self.hparams.dataset_name:
                m = torch.nn.functional.pad(m, (3, 2, -7, -6, 3, 2), value=background_value) # [96,96,96]
            elif self.hparams.dataset_name == "ABCD":
                m = torch.nn.functional.pad(m, (6, -5, -11, -10, -1, -2), value=background_value) # [96,96,96]
            else:
                # Default: assume already [96,96,96]
                if m.shape != (96,96,96):
                    raise ValueError(f"Unexpected mask shape {tuple(m.shape)}; expect [96,96,96]")

            # Binarize
            m = (m > 0)

            # Register MNI152_mask and downsampled version
            self.register_buffer("MNI152_mask", m)  # [96,96,96] bool
            with torch.no_grad():
                _m = m[None, None].float()  # [1,1,96,96,96]
                _m_ds = F.interpolate(_m, size=(6, 6, 6), mode="nearest")[0, 0].to(torch.bool)
            self.register_buffer("MNI152_mask_ds", _m_ds)                  # [6,6,6]
            self.register_buffer("MNI152_mask_ds_flat", _m_ds.reshape(-1)) # [216]
        else:
            m = torch.ones((96,96,96), dtype=torch.bool)
            self.register_buffer("MNI152_mask", m)
            with torch.no_grad():
                _m = m[None, None].float()
                _m_ds = F.interpolate(_m, size=(6, 6, 6), mode="nearest")[0, 0].to(torch.bool)
            self.register_buffer("MNI152_mask_ds", _m_ds)
            self.register_buffer("MNI152_mask_ds_flat", _m_ds.reshape(-1))

        if self.hparams.adjust_thresh:
            self.threshold = 0


        # ===== ROI atlas for FC metrics (rfMRI_next) =====
        self.use_fc_metrics = bool(getattr(self.hparams, "use_fc_metrics", False))

        self.atlas_pt = getattr(self.hparams, "roi_atlas_pt", None)
        if self.use_fc_metrics and self.atlas_pt is not None:
            atlas = torch.load(self.atlas_pt, map_location="cpu").long()  # [96,96,96]
            if hasattr(self, "MNI152_mask") and (self.MNI152_mask is not None):
                atlas = atlas * self.MNI152_mask.long()
            self.register_buffer("roi_atlas_96", atlas)
        
            roi_ids = torch.unique(atlas)
            roi_ids = roi_ids[roi_ids > 0]
            roi_ids, _ = torch.sort(roi_ids)
            self.register_buffer("roi_ids", roi_ids)

            # cache voxel indices per ROI for speed
            roi_voxel_indices = []
            atlas_flat = atlas.view(-1)
            for rid in roi_ids.tolist():
                idx = torch.nonzero(atlas_flat == rid, as_tuple=False).squeeze(1)  # [n_vox_r]
                roi_voxel_indices.append(idx)
            self.roi_voxel_indices = roi_voxel_indices   # Python list[Tensor], 放在CPU也行

        else:
            self.roi_atlas_96 = None
            self.roi_ids = None
            self.roi_voxel_indices = None 



    def forward(self, x):
        return self.output_head(self.model(x))
    
    def augment(self, img):
        if self.hparams.downstream_task == 'tfMRI_3D':
            B, T, H, W, D = img.shape

            device = img.device

            rand_affine = monai_t.RandAffine(
                prob=1.0,
                # 0.175 rad = 10 degrees
                rotate_range=(0.175, 0.175, 0.175),
                scale_range = (0.1, 0.1, 0.1),
                mode = "bilinear",
                padding_mode = "border",
                device = device
            )

            rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
            rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
            if self.hparams.augment_only_intensity:
                comp = monai_t.Compose([rand_noise, rand_smooth])
            else:
                comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

            for b in range(B):
                aug_seed = torch.randint(0, 10000000, (1,)).item()
                # set augmentation seed to be the same for all time steps
                for t in range(T):
                    if self.hparams.augment_only_affine:
                        rand_affine.set_random_state(seed=aug_seed)
                        img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                    else:
                        comp.set_random_state(seed=aug_seed)
                        img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        else:
            B, C, H, W, D, T = img.shape

            device = img.device
            # print("img device: ", img.device)
            img = rearrange(img, 'b c h w d t -> b t c h w d')
            # print("img shape: ", img.shape)

            rand_affine = monai_t.RandAffine(
                prob=1.0,
                # 0.175 rad = 10 degrees
                rotate_range=(0.175, 0.175, 0.175),
                scale_range = (0.1, 0.1, 0.1),
                mode = "bilinear",
                padding_mode = "border",
                device = device
            )
            rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
            rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) # 

            for b in range(B):
                aug_seed = torch.randint(0, 10000000, (1,)).item()
                # set augmentation seed to be the same for all time steps
                for t in range(T):
                    comp.set_random_state(seed=aug_seed)
                    # print("input shape: ", img[b, t, :, :, :, :].shape)
                    # aug_img = comp(img[b, t, :, :, :, :])
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

                    # rand_affine.set_random_state(seed=aug_seed)
                    # img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])

            img = rearrange(img, 'b t c h w d -> b c h w d t')
            
        return img

    def _to_voxel_logits(self, x):
        """
        Upsample model outputs (logits) of arbitrary shape to [B, 96, 96, 96]
        Steps: remove temporal dimension -> reduce channels to 1 -> upsample to 96³
        """
        target_spatial = tuple(self.hparams.img_size[:3])  # (96,96,96)

        # Remove temporal dimension
        if x.ndim == 6:                # [B, C, D', H', W', T] 或 [B, D', H', W', T]
            x = x.mean(dim=-1)
        if x.ndim == 4:                # [B, D', H', W'] -> [B,1,D',H',W']
            x = x.unsqueeze(1)
        assert x.ndim == 5, f"Expect 5D [B,C,D,H,W], got {tuple(x.shape)}"

        B, C, Dp, Hp, Wp = x.shape

        # Reduce channels to 1
        if C > 1:
            if not hasattr(self, "_proj3d"):
                self._proj3d = torch.nn.Conv3d(C, 1, kernel_size=1, bias=True).to(x.device)
            x = self._proj3d(x)        # [B,1,D',H',W']

        # Upsample to 96³
        if (Dp, Hp, Wp) != target_spatial:
            x = F.interpolate(x, size=target_spatial, mode="trilinear", align_corners=False)  # [B,1,96,96,96]

        return x.squeeze(1)            # [B,96,96,96]
    
    def _project_to_voxels_time(self, x, out_T=None):
        """
        Map arbitrary-shaped network outputs to [B, 1, 96, 96, 96, T].
        Supports x of shapes: [B, C, D', H', W', T] / [B, D', H', W', T] / [B, C, D', H', W'].
        out_T: Target temporal length; if None, keep the original T.
        """
        target_spatial = tuple(self.hparams.img_size[:3])  # (96, 96, 96)

        # Normalize to [B, C, D, H, W, T]
        if x.ndim == 4:                     # [B, D, H, W]
            x = x.unsqueeze(1).unsqueeze(-1)  # -> [B,1,D,H,W,1]
        elif x.ndim == 5:
            plausible_c = {1,2,3,4,6,8,12,16}
            if x.shape[1] in plausible_c:     # [B,C,D,H,W]
                x = x.unsqueeze(-1)           # -> [B,C,D,H,W,1]
            else:                             # 认为是 [B,D,H,W,T]
                x = x.unsqueeze(1)            # -> [B,1,D,H,W,T]
        else:
            assert x.ndim == 6, f"expect 6D [B,C,D,H,W,T], got {tuple(x.shape)}"
        B, C, Dp, Hp, Wp, T = x.shape

        # Project channels to 1 (independently for each time frame)
        if C > 1:
            if not hasattr(self, "_proj3d_time"):
                self._proj3d_time = torch.nn.Conv3d(C, 1, kernel_size=1, bias=True).to(x.device)
            x = torch.stack([self._proj3d_time(x[..., t]) for t in range(T)], dim=-1)  # [B, 1, D', H', W', T]

        # Upsample spatially to 96^3 (frame by frame)
        if (Dp, Hp, Wp) != target_spatial:
            x = torch.stack([
                F.interpolate(x[..., t], size=target_spatial, mode="trilinear", align_corners=False)
                for t in range(T)
            ], dim=-1)   # [B, 1, 96, 96, 96, T]

        # Align temporal length (truncate/repeat if needed; ideally have the backbone produce correct T)
        if out_T is not None and T != out_T:
            if T > out_T:
                x = x[..., :out_T]
            else:
                x = torch.cat([x, x[..., -1:].repeat(1, 1, 1, 1, 1, out_T - T)], dim=-1)

        return x  # [B, 1, 96, 96, 96, out_T or T]

    @torch.no_grad()
    def _rollout_log_k_curves(self, batch):
        if self.hparams.k_max is None or self.hparams.k_max <= 0: 
            return
        fmri, subj, *_ = batch.values()   # [B,1,H,W,D,T_full]
        B, C, H, W, D, T_full = fmri.shape

        # Decide T_ctx
        if self.hparams.pred_context is None:
            T_ctx = T_full // 2
        else:
            T_ctx = int(self.hparams.pred_context)

        K = min(self.hparams.k_max, T_full - T_ctx - 1)
        if K <= 0: 
            return

        # Initial context (ground truth)
        ctx = fmri[..., :T_ctx].detach()          # [B,1,H,W,D,T_ctx]

        for k in range(1, K+1):
            # Predict the next frame (strictly use horizon=1)
            feat_next = self.model(ctx)
            pred_next_raw = self._project_to_voxels_time(self.output_head(feat_next), out_T=1) # [B,1,96,96,96,1]

            if getattr(self.hparams, "use_residual_branch", False) and (self.residual_head is not None):
                res_raw  = self._project_to_voxels_time(self.residual_head(feat_next), out_T=1)
                base_raw = self._rfmri_baseline_one_step(ctx, getattr(self.hparams, "residual_base", "persist"), apply_mask=False)
                pred_next_raw = base_raw + res_raw  
                
            
            gt_next   = self._project_to_voxels_time(fmri[..., T_ctx + k - 1:T_ctx + k], out_T=1)

            if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                m = self.MNI152_mask[None, None, ...].to(pred_next_raw.dtype).unsqueeze(-1)  # [1,1,96,96,96,1]
                pred_next = pred_next_raw * m
                gt_next   = gt_next   * m
            else:
                pred_next = pred_next_raw
            
            # Compute mse@k / r@k
            # MSE@k (average only within brain region)
            m6 = self.MNI152_mask[None, None, ...].unsqueeze(-1).float()\
                    .to(pred_next.device).expand_as(pred_next)          # [B,1,96,96,96,1]
            se = (pred_next - gt_next).pow(2)
            mse_k = (se * m6).sum() / m6.sum().clamp_min(1.0)

            # r@k: Pearson correlation along spatial dimensions (only within brain region)
            pred_sp = pred_next.squeeze(1).squeeze(-1)                  # [B,96,96,96]
            gt_sp   = gt_next.squeeze(1).squeeze(-1)                    # [B,96,96,96]
            mask_flat = self.MNI152_mask.view(-1).bool()
            pred_f = pred_sp.view(B, -1)[:, mask_flat]                  # [B, V_mask]
            gt_f   = gt_sp.view(B, -1)[:, mask_flat]                    # [B, V_mask]
            x = pred_f - pred_f.mean(dim=1, keepdim=True)
            y = gt_f   - gt_f.mean(dim=1, keepdim=True)
            r_k = (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-6)
            r_k = r_k.mean()

            self.log(f"valid_mse@{k}", mse_k, prog_bar=False, sync_dist=False, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
            self.log(f"valid_r@{k}",   r_k,   prog_bar=False, sync_dist=False, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

            # Sliding window: drop the leftmost frame and append the prediction
            ctx = torch.cat([ctx[..., 1:], pred_next_raw], dim=-1)

    
    @staticmethod
    def _r_time_mean(x, y, mask=None):
        """Mean Pearson correlation along time; supports [B,1,D,H,W,T] or [B,V,T].
           If mask is provided (96x96x96 bool), only voxels inside mask are used.
        """
        def to_BVT(t):
            if t.ndim == 6:
                return t.view(t.shape[0], -1, t.shape[-1])
            if t.ndim == 3:
                return t
            raise AssertionError(f"expect [B,1,D,H,W,T] or [B,V,T], got {tuple(t.shape)}")

        X = to_BVT(x)
        Y = to_BVT(y)
        if mask is not None:
            m = mask.view(-1).bool()
            X = X[:, m, :]
            Y = Y[:, m, :]

        Xz = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-6)
        Yz = (Y - Y.mean(-1, keepdim=True)) / (Y.std(-1, keepdim=True) + 1e-6)
        return (Xz * Yz).mean()

    def _voxel_to_roi_ts(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,96,96,96,T]
        return: [B, N_roi, T]
        """
        assert self.roi_atlas_96 is not None and self.roi_ids is not None
        B, C, H, W, D, T = x.shape
        assert C == 1 and (H,W,D) == (96,96,96)
    
        x_flat = x.view(B, V, T)                        # [B,V,T]
    
        # compute ROI mean time series
        roi_ts_list = []
        for idx in self.roi_voxel_indices:
            idx = idx.to(x_flat.device)
            roi_ts_list.append(x_flat.index_select(1, idx).mean(dim=1))  # [B,T]
    
        return torch.stack(roi_ts_list, dim=1)  # [B,N,T]
    
    def _roi_ts_to_fc_vec(self, roi_ts: torch.Tensor) -> torch.Tensor:
        """
        roi_ts: [B,N,T]
        return: [B, N*(N-1)/2]  (upper triangle)
        """
        B, N, T = roi_ts.shape
        z = roi_ts - roi_ts.mean(dim=-1, keepdim=True)
        z = z / (roi_ts.std(dim=-1, keepdim=True) + 1e-6)
    
        fc = torch.einsum("bnt,bmt->bnm", z, z) / float(T)  # [B,N,N]
    
        iu = torch.triu_indices(N, N, offset=1, device=fc.device)
        vec = fc[:, iu[0], iu[1]]                           # [B, F]
        return vec

    
    @torch.no_grad()
    def _log_rfMRI_baselines(self, batch, mode="valid"):
        # Fetch fmri
        fmri = batch["fmri_sequence"] if isinstance(batch, dict) and "fmri_sequence" in batch else list(batch.values())[0]
        B, C, H, W, D, T_full = fmri.shape

        # Split into context / future
        T_ctx = T_full // 2 if (self.hparams.pred_context is None or self.hparams.pred_horizon is None) \
            else int(self.hparams.pred_context)
        T_hz  = (T_full - T_ctx) if (self.hparams.pred_context is None or self.hparams.pred_horizon is None) \
            else int(self.hparams.pred_horizon)
        if T_ctx <= 0 or T_hz <= 0 or T_ctx + T_hz > T_full:
            return

        x_ctx    = fmri[..., :T_ctx]                 # [B,1,96,96,96,T_ctx]
        y_future = fmri[..., T_ctx:T_ctx+T_hz]       # [B,1,96,96,96,T_hz]

        # Two baselines
        base_persist = x_ctx[..., -1:].repeat(1,1,1,1,1,T_hz)   # persistence
        base_mean    = x_ctx.mean(dim=-1, keepdim=True).repeat(1,1,1,1,1,T_hz)  # context mean
        pf_raw = base_persist.flatten(start_dim=1)
        mf_raw = base_mean.flatten(start_dim=1)
        yf_raw = y_future.flatten(start_dim=1)

        mse_p_raw = F.mse_loss(pf_raw, yf_raw)
        mae_p_raw = F.l1_loss(pf_raw, yf_raw)
        mse_m_raw = F.mse_loss(mf_raw, yf_raw)
        mae_m_raw = F.l1_loss(mf_raw, yf_raw)

        r_p_raw   = self._r_time_mean(base_persist, y_future)
        r_m_raw   = self._r_time_mean(base_mean,    y_future)

        self.log(f"{mode}_mse_baseline_persist_nomask", mse_p_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_mae_baseline_persist_nomask", mae_p_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_r_time_baseline_persist_nomask", r_p_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

        self.log(f"{mode}_mse_baseline_ctxmean_nomask", mse_m_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_mae_baseline_ctxmean_nomask", mae_m_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_r_time_baseline_ctxmean_nomask", r_m_raw, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)


        # Apply mask if available
        if not hasattr(self, "MNI152_mask") or self.MNI152_mask is None:
            return  

        m6 = self.MNI152_mask[None, None, ...].unsqueeze(-1).float().to(y_future.device).expand_as(y_future)  # [B,1,96,96,96,T_hz]

        def mse_mask(a, b):
            se = (a - b).pow(2) * m6
            return se.sum() / m6.sum().clamp_min(1.0)

        def mae_mask(a, b):
            ae = (a - b).abs() * m6
            return ae.sum() / m6.sum().clamp_min(1.0)

        mse_p = mse_mask(base_persist, y_future)
        mae_p = mae_mask(base_persist, y_future)
        mse_m = mse_mask(base_mean,    y_future)
        mae_m = mae_mask(base_mean,    y_future)

        r_p = self._r_time_mean(base_persist, y_future, mask=self.MNI152_mask)
        r_m = self._r_time_mean(base_mean,    y_future, mask=self.MNI152_mask)
        
        #check
        with torch.no_grad():
            y_std = torch.sqrt(((y_future**2) * m6).sum() / m6.sum().clamp_min(1.0))
            rmse_p = torch.sqrt(((base_persist - y_future).pow(2) * m6).sum() / m6.sum().clamp_min(1.0))
            rel_rmse_p = rmse_p / (y_std + 1e-8)
            rel_rmse_expected = torch.sqrt(torch.clamp(2 * (1.0 - r_p), min=0.0))
        self.log(f"{mode}_y_std_mask", y_std, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_rmse_baseline_persist", rmse_p, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_relrmse_persist", rel_rmse_p, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_relrmse_expected_from_rho1", rel_rmse_expected, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

        self.log(f"{mode}_mse_baseline_persist", mse_p, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_mae_baseline_persist", mae_p, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_r_time_baseline_persist", r_p, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

        self.log(f"{mode}_mse_baseline_ctxmean",  mse_m, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_mae_baseline_ctxmean",  mae_m, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_r_time_baseline_ctxmean", r_m, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log(f"{mode}_mask_fraction", self.MNI152_mask.float().mean(), on_step=False, on_epoch=True)

    def _tf_ratio(self):
        if not self.training:
            return 0.0
        base = float(getattr(self.hparams, "teacher_forcing_ratio", 0.0))
        if base <= 0.0:
            return 0.0  

        schedule = getattr(self.hparams, "tf_schedule", "linear")
        T = max(1, (self.trainer.max_epochs or 1))
        e = float(self.current_epoch)

        if schedule == "constant":
            return base
        if schedule == "linear":
            return base * max(0.0, 1.0 - e / (0.8 * T))
        if schedule == "cosine":
            import math
            x = min(e / (0.8 * T), 1.0)
            return base * 0.5 * (1.0 + math.cos(math.pi * x))
        if schedule == "exp":
            return base * (0.95 ** e)
        return base

    
    def _rfmri_baseline_one_step(self, ctx, kind: str, apply_mask: bool = True):
        """
        ctx: [B,1,96,96,96,T_ctx]
        return: baseline next-frame [B,1,96,96,96,1]
        """
        if kind == "persist":
            base = ctx[..., -1:]
        elif kind == "ctxmean":
            base = ctx.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown residual_base={kind}")
    
        if apply_mask and hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
            m = self.MNI152_mask[None, None, ...].to(base.dtype).unsqueeze(-1)
            base = base * m
        return base

    
    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training:
            fmri = self.augment(fmri)
        
        # ===== rfMRI next-token =====
        if getattr(self.hparams, "downstream_task", "") == "rfMRI_next":
            # fmri: [B,1,H,W,D,T_full]
            # fmri = batch["fmri_sequence"]
            B, C, H, W, D, T_full = fmri.shape

            # context & horizon
            if self.hparams.pred_context is None or self.hparams.pred_horizon is None:
                T_ctx = T_full // 2
                T_hz  = T_full - T_ctx
            else:
                T_ctx = int(self.hparams.pred_context)
                T_hz  = int(self.hparams.pred_horizon)
            assert T_ctx + T_hz <= T_full, "pred_context + pred_horizon > T_full"

            ctx = fmri[..., :T_ctx]         
            # [B,1,96,96,96,T_ctx]
            y_future = fmri[..., T_ctx:T_ctx+T_hz]     # [B,1,96,96,96,T_hz]
            tf_ratio = self._tf_ratio()

            
            preds_main = []
            preds_res  = [] if (self.residual_head is not None) else None
            for t in range(T_hz):
                feat_t = self.model(ctx)
            
                # ===== main branch (unchanged semantics) =====
                pred_t_raw = self._project_to_voxels_time(self.output_head(feat_t), out_T=1)  # [B,1,96,96,96,1]
                # mask (if any)
                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    m = self.MNI152_mask[None, None, ...].to(pred_t_raw.dtype).unsqueeze(-1)
                    pred_t_masked = pred_t_raw * m
                else:
                    pred_t_masked = pred_t_raw
            
                preds_main.append(pred_t_raw)
            
                # ========= residual branch ==========
                pred_res_raw = None
                pred_res_masked = None
                if self.residual_head is not None:
                    # residual predicted by head (RAW, no mask, so it can affect ctx)
                    res_raw = self._project_to_voxels_time(self.residual_head(feat_t), out_T=1)  # [B,1,96,96,96,1]
            
                    # baseline (RAW, no mask, so rollout semantics is clean)
                    base_raw = self._rfmri_baseline_one_step(ctx, getattr(self.hparams, "residual_base", "persist"), apply_mask=False)
                    pred_res_raw = base_raw + res_raw  # fed back to ctx
            
                    # masked version only for loss/metric logging
                    if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                        m = self.MNI152_mask[None, None, ...].to(pred_res_raw.dtype).unsqueeze(-1)
                        pred_res_masked = pred_res_raw * m
                    else:
                        pred_res_masked = pred_res_raw
            
                    preds_res.append(pred_res_masked)

            
                # ===== teacher forcing update =====
                use_teacher = (torch.rand((), device=ctx.device) < tf_ratio).item()
                if use_teacher:
                    next_frame = y_future[..., t:t+1]     # GT
                else:
                    if pred_res_raw is not None:
                        next_frame = pred_res_raw         # residual AR
                    else:
                        next_frame = pred_t_raw           # main AR

                ctx = torch.cat([ctx, next_frame], dim=-1)
                if getattr(self.hparams, "ctx_update", "sliding") == "sliding":
                    # keep fixed length T_ctx
                    if ctx.shape[-1] > T_ctx:
                        ctx = ctx[..., -T_ctx:]
                elif self.hparams.ctx_update == "growing":
                    # do nothing, let it grow
                    pass
                else:
                    raise ValueError(f"Unknown ctx_update={self.hparams.ctx_update}")


            logits_main = torch.cat(preds_main, dim=-1)          # [B,1,96,96,96,T_hz]
            target = y_future.clone()
            #only use mask in loss/metric, don't use in the rollout
            # if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
            #     m = self.MNI152_mask[None, None, ...].to(target.dtype).unsqueeze(-1).expand_as(target)
            #     target = target * m

            # subj = batch["subject_name"]

            # stash residual logits for loss
            if preds_res is not None:
                self._last_rfMRI_logits_res = torch.cat(preds_res, dim=-1)  # [B,1,96,96,96,T_hz]
            else:
                self._last_rfMRI_logits_res = None
            
            return subj, logits_main, target


        # ===== tfMRI =====
        feature = self.model(fmri)
        if self.hparams.downstream_task == 'tfMRI_3D':
            # logits = self.output_head(feature) 
            logits = self._to_voxel_logits(self.output_head(feature))   # [B,96,96,96]
            logits = logits * self.MNI152_mask  
            target = target_value.float() * self.MNI152_mask    # [B,96,96,96]
            #logits = logits[:,(padded_mask > 0 ).expand(logits.size()[1:])]
            # target = target_value.float().squeeze() * self.MNI152_mask # 96, 96, 96: masked image
        # Classification task
        elif self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            logits = self.output_head(feature).squeeze() #self.clf(feature).squeeze()
            target = target_value.float().squeeze()
        # Regression task
        elif self.hparams.downstream_task_type == 'regression':
            # target_mean, target_std = self.determine_target_mean_std()
            logits = self.output_head(feature) # (batch,1) or # tuple((batch,1), (batch,1))
            unnormalized_target = target_value.float() # (batch,1)
            if self.hparams.label_scaling_method == 'standardization': # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        
        return subj, logits, target

    
    
    def _calculate_loss(self, batch, batch_idx, mode):
        subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training)
        
        #=== rfMRI ===
        if getattr(self.hparams, "downstream_task", "") == "rfMRI_next":
             
            # [B,1,96,96,96,T] -> [B, V*T]
            logits_f = logits.flatten(start_dim=1)
            target_f = target.flatten(start_dim=1)

            use_fc = bool(getattr(self.hparams, "use_fc_metrics", False)) and (self.roi_atlas_96 is not None)
            if use_fc:
                # logits/target: [B,1,96,96,96,T] -> FC vec [B,F]
                roi_ts_pred = self._voxel_to_roi_ts(logits)
                roi_ts_gt   = self._voxel_to_roi_ts(target)
                feat_pred   = self._roi_ts_to_fc_vec(roi_ts_pred)  # [B,F]
                feat_gt     = self._roi_ts_to_fc_vec(roi_ts_gt)    # [B,F]

                # ===== loss on FC vec =====
                if self.hparams.loss_type == "mae":
                    loss = F.l1_loss(feat_pred, feat_gt)
                    within_subj_loss = loss
                    across_subj_loss = torch.zeros((), device=logits.device)
            
                elif self.hparams.loss_type == "rc":
                    within_subj_loss, across_subj_loss = contrast_mse_loss(feat_pred, feat_gt, subj)
                    loss = (self.hparams.within_subj_margin * within_subj_loss
                            - self.hparams.across_subj_margin * across_subj_loss)
            
                else:  # mse
                    loss = F.mse_loss(feat_pred, feat_gt)
                    within_subj_loss = loss
                    across_subj_loss = torch.zeros((), device=logits.device)
            
            else:
                # ===== original voxel loss =====
                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    m6 = (
                        self.MNI152_mask[None, None, ...]
                        .unsqueeze(-1)
                        .float()
                        .to(logits.device)
                        .expand_as(logits)
                    )  # [B,1,96,96,96,T]
                else:
                    m6 = torch.ones_like(logits, dtype=torch.float32)
    
                if self.hparams.loss_type == "mae":
                    # MAE averaged within the mask
                    loss = ((logits - target).abs() * m6).sum() / m6.sum().clamp_min(1.0)
                    within_subj_loss = loss
                    across_subj_loss = torch.zeros(1, device=logits.device)
                elif self.hparams.loss_type == "rc":
                    # Apply mask and then flatten (keep only brain-region voxels)
                    B, _, _, _, _, T = logits.shape
                    if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                        mask_flat = self.MNI152_mask.view(-1).bool()
                        L   = logits.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                        Tgt = target.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                    else:
                        L   = logits.flatten(start_dim=1)
                        Tgt = target.flatten(start_dim=1)
                    within_subj_loss, across_subj_loss = contrast_mse_loss(L, Tgt, subj)
                    loss = (
                        self.hparams.within_subj_margin * within_subj_loss
                        - self.hparams.across_subj_margin * across_subj_loss
                    )
                else:  # MSE averaged within the mask
                    loss = ((logits - target).pow(2) * m6).sum() / m6.sum().clamp_min(1.0)
                    within_subj_loss = loss
                    across_subj_loss = torch.zeros(1, device=logits.device)


            # Log r (temporal correlation): treat [B,1,96,96,96,T] as [B, V, T]
            with torch.no_grad():
                if use_fc:
                    mse_val = F.mse_loss(feat_pred, feat_gt)
                    l1_val  = F.l1_loss(feat_pred, feat_gt)
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_mse": mse_val,
                        f"{mode}_l1_loss": l1_val,
                    }
                else:
                    if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                        r_time_mean = self._r_time_mean(logits, target, mask=self.MNI152_mask)
                    else:
                        r_time_mean = self._r_time_mean(logits, target)
                    mse_val = ((logits - target).pow(2) * m6).sum() / m6.sum().clamp_min(1.0)
                    l1_val  = ((logits - target).abs()   * m6).sum() / m6.sum().clamp_min(1.0)
    
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_mse": mse_val,
                        f"{mode}_l1_loss": l1_val,
                        f"{mode}_r_time_mean": r_time_mean,
                    }
            self.log_dict(
                result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False,
                on_step=True, on_epoch=True, batch_size=self.hparams.batch_size
            )

            #============= residual loss ==============
            if getattr(self.hparams, "use_residual_branch", False):
                logits_res = getattr(self, "_last_rfMRI_logits_res", None)
                if logits_res is not None:
                    with torch.no_grad():
                        if use_fc:
                            roi_ts_res = self._voxel_to_roi_ts(logits_res)
                            feat_res   = self._roi_ts_to_fc_vec(roi_ts_res)
                            mse_res = F.mse_loss(feat_res, feat_gt)
                            l1_res  = F.l1_loss(feat_res, feat_gt)
                            self.log_dict(
                                {f"{mode}_mse_residual_branch": mse_res,
                                 f"{mode}_l1_residual_branch":  l1_res},
                                on_step=False, on_epoch=True, prog_bar=False,
                                batch_size=self.hparams.batch_size
                            )
                            w = float(getattr(self.hparams, "residual_l2", 0.0))
                            if w > 0:
                                loss = loss + w * mse_res
                                self.log(f"{mode}_loss_residual_branch", mse_res,
                                         on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
                        else:
                            mse_res = ((logits_res - target).pow(2) * m6).sum() / m6.sum().clamp_min(1.0)
                            l1_res  = ((logits_res - target).abs()   * m6).sum() / m6.sum().clamp_min(1.0)
                            if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                                r_res = self._r_time_mean(logits_res, target, mask=self.MNI152_mask)
                            else:
                                r_res = self._r_time_mean(logits_res, target)
    
                        self.log_dict(
                            {
                                f"{mode}_mse_residual_branch": mse_res,
                                f"{mode}_l1_residual_branch": l1_res,
                                f"{mode}_r_time_residual_branch": r_res,
                            },
                            on_step=False, on_epoch=True, prog_bar=False,
                            batch_size=self.hparams.batch_size
                        )
    
                        # optional：add residual loss into main loss（ 0 by default only for slog）
                        w = float(getattr(self.hparams, "residual_l2", 0.0))
                        if w > 0:
                            loss_res = ((logits_res - target).pow(2) * m6).sum() / m6.sum().clamp_min(1.0)
                            loss = loss + w * loss_res
                            self.log(f"{mode}_loss_residual_branch", loss_res,
                                     on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

            return loss

        #=== tfMRI ===
        if 'tfMRI' in self.hparams.downstream_task:
            B = logits.shape[0]
            if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                mask_flat = self.MNI152_mask.view(-1).bool()
                logits = logits.view(B, -1)[:, mask_flat]
                target = target.view(B, -1)[:, mask_flat]
            else:
                logits = logits.flatten(start_dim=1)
                target = target.flatten(start_dim=1)
                
            if self.hparams.loss_type == 'mse':
                logits = logits.flatten(start_dim=1) # (batch,mask_dim)
                target = target.flatten(start_dim=1) # (batch,mask_dim)
                loss = F.mse_loss(logits, target)
                within_subj_loss = loss
                across_subj_loss = torch.zeros(1)
            elif self.hparams.loss_type == 'mae':
                logits = logits.flatten(start_dim=1) # (batch,mask_dim)
                target = target.flatten(start_dim=1) # (batch,mask_dim)
                loss = F.l1_loss(logits, target)
                within_subj_loss = loss
                across_subj_loss = torch.zeros(1)
            elif self.hparams.loss_type == 'rc':
                logits = logits.flatten(start_dim=1) # (batch,mask_dim)
                target = target.flatten(start_dim=1) # (batch,mask_dim)
                within_subj_loss, across_subj_loss = contrast_mse_loss(logits, target, subj)
                loss = self.hparams.within_subj_margin * within_subj_loss - self.hparams.across_subj_margin * across_subj_loss

            mse = F.mse_loss(logits, target)
            l1 = F.l1_loss(logits, target)
            result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_mse": mse, 
                f"{mode}_l1_loss": l1,
                f"{mode}_within_subj_loss": within_subj_loss.item(),
                f"{mode}_across_subj_loss": across_subj_loss.item(),
            }
        elif self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            loss = F.binary_cross_entropy_with_logits(logits, target) # target is float
            acc = self.metric.get_accuracy_binary(logits, target.float().squeeze())
            result_dict = {
            f"{mode}_loss": loss,
            f"{mode}_acc": acc,
            }

        elif self.hparams.downstream_task_type == 'regression':
            loss = F.mse_loss(logits.squeeze(), target.squeeze())
            l1 = F.l1_loss(logits.squeeze(), target.squeeze())
            result_dict = {
                f"{mode}_loss": loss,
                f"{mode}_mse": loss,
                f"{mode}_l1_loss": l1
            }
            
        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size) # batch_size = batch_size
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode):
        """
        total_out: torch.Tensor on CPU or GPU
          - tfMRI / rfMRI_next(voxel): [N_samples, F, 2]
          - rfMRI_next(FC vec):        [N_samples, F, 2]  (F = N_roi*(N_roi-1)/2)
        subj_array: np.array of subject ids, length N_samples
        """
        task = getattr(self.hparams, "downstream_task", "")
        subjects = np.unique(subj_array)
    
        def aggregate_subject_level(get_gt_mode: str):
            """
            get_gt_mode:
              - "first": GT per subject assumed constant across segments
              - "mean":  GT per subject varies across segments, average it
            """
            device = total_out.device
            Fdim = total_out.shape[1]
            subj_pred = torch.empty((len(subjects), Fdim), device=device)
            subj_gt   = torch.empty((len(subjects), Fdim), device=device)
    
            for si, s in enumerate(subjects):
                idxs = np.where(subj_array == s)[0]
                pred_seg = total_out[idxs, :, 0]  # [n_seg, F]
                gt_seg   = total_out[idxs, :, 1]  # [n_seg, F]
    
                subj_pred[si] = pred_seg.mean(dim=0)
    
                if get_gt_mode == "first":
                    subj_gt[si] = gt_seg[0]
                elif get_gt_mode == "mean":
                    subj_gt[si] = gt_seg.mean(dim=0)
                else:
                    raise ValueError(f"Unknown get_gt_mode={get_gt_mode}")
    
            return subj_pred, subj_gt
    
        if task == "rfMRI_next":
            # GT mean
            subj_avg_logits, subj_targets = aggregate_subject_level(get_gt_mode="mean")
    
        elif "tfMRI" in task:
            # GT first
            subj_avg_logits, subj_targets = aggregate_subject_level(get_gt_mode="first")

        else:
            # meta data prediction
            subj_avg_logits = []
            subj_targets = []
            for subj in subjects:
                subj_logits = total_out[subj_array == subj,0] 
                subj_avg_logits.append(torch.mean(subj_logits).item())
                subj_targets.append(total_out[subj_array == subj,1][0].item())
            subj_avg_logits = torch.tensor(subj_avg_logits, device = total_out.device) 
            subj_targets = torch.tensor(subj_targets, device = total_out.device) 
        
            if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                if self.hparams.adjust_thresh:
                    # move threshold to maximize balanced accuracy
                    best_bal_acc = 0
                    best_thresh = 0
                    for thresh in np.arange(-5, 5, 0.01):
                        bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=thresh).int().cpu())
                        if bal_acc > best_bal_acc:
                            best_bal_acc = bal_acc
                            best_thresh = thresh
                    self.log(f"{mode}_best_thresh", best_thresh, sync_dist=True)
                    self.log(f"{mode}_best_balacc", best_bal_acc, sync_dist=True)
                    fpr, tpr, thresholds = roc_curve(subj_targets.cpu(), subj_avg_logits.cpu())
                    idx = np.argmax(tpr - fpr)
                    youden_thresh = thresholds[idx]
                    acc_func = BinaryAccuracy().to(total_out.device)
                    self.log(f"{mode}_youden_thresh", youden_thresh, sync_dist=True)
                    self.log(f"{mode}_youden_balacc", balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=youden_thresh).int().cpu()), sync_dist=True)

                    if mode == 'valid':
                        self.threshold = youden_thresh
                    elif mode == 'test':
                        bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=self.threshold).int().cpu())
                        self.log(f"{mode}_balacc_from_valid_thresh", bal_acc, sync_dist=True)
                else:
                    acc_func = BinaryAccuracy().to(total_out.device)

                auroc_func = BinaryAUROC().to(total_out.device)
                acc = acc_func((subj_avg_logits >= 0).int(), subj_targets)
                #print((subj_avg_logits>=0).int().cpu())
                #print(subj_targets.cpu())
                bal_acc_sk = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=0).int().cpu())
                auroc = auroc_func(torch.sigmoid(subj_avg_logits), subj_targets)
                self.log(f"{mode}_acc", acc, sync_dist=True)
                self.log(f"{mode}_balacc", bal_acc_sk, sync_dist=True)
                self.log(f"{mode}_AUROC", auroc, sync_dist=True)

            # regression target is normalized
            elif self.hparams.downstream_task_type == 'regression':          
                mse = F.mse_loss(subj_avg_logits, subj_targets)
                mae = F.l1_loss(subj_avg_logits, subj_targets)

                # reconstruct to original scale
                if self.hparams.label_scaling_method == 'standardization': # default
                    adjusted_mse = F.mse_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                    adjusted_mae = F.l1_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                elif self.hparams.label_scaling_method == 'minmax':
                    adjusted_mse = F.mse_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                    adjusted_mae = F.l1_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                pearson = PearsonCorrCoef().to(total_out.device)
                prearson_coef = pearson(subj_avg_logits, subj_targets)
                
                r2score = R2Score()
                r2_output = r2score(subj_avg_logits, subj_targets)

                self.log(f"{mode}_corrcoef", prearson_coef, sync_dist=True)
                self.log(f"{mode}_r2", r2_output, sync_dist=True)
                self.log(f"{mode}_mse", mse, sync_dist=True)
                self.log(f"{mode}_mae", mae, sync_dist=True)
                self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True) 
                self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True) 
            return
    
        # ===== shared vector metrics for tfMRI / rfMRI_next =====
        mse = F.mse_loss(subj_avg_logits, subj_targets)
        mae = F.l1_loss(subj_avg_logits, subj_targets)
        if subj_avg_logits.ndim == 1 : # voxels (only)
            A = subj_avg_logits.unsqueeze(1)
            B = subj_targets.unsqueeze(1)
        elif subj_avg_logits.ndim == 2 :
            A = torch.transpose(subj_avg_logits,0,1) # voxels * subjects
            B = torch.transpose(subj_targets,0,1)
        else:
            print(f'the dimension of target and logits are not as expected:{subj_avg_logits.ndim}')
        # assumes verticesXsubject matrices, returns subjectXsubject corrmat 
        A = (A - A.mean(axis=0)) / A.std(axis=0)
        B = (B - B.mean(axis=0)) / B.std(axis=0)
        
        corrmat = torch.einsum("ik,kj->ij",B.t(), A) / B.shape[0] 
        #print('corrmat',corrmat) 
        
        corrmat = corrmat.detach().cpu()
        #top1_diag_acc
        top1_diag_acc = torch.mean((torch.argmax(corrmat,axis=1) == torch.arange(corrmat.shape[0])).float()).item()
        
        #diagonality_rank_score
        sorted_tensor = torch.argsort(corrmat,axis=1)
        diag_rank_idx = torch.mean(torch.tensor([torch.where(sorted_tensor[i,:]==i )[0] / (corrmat.shape[0]-1) for i in range(corrmat.shape[0])]).float()).item()
    
        diag = torch.einsum('ii->i', corrmat) # torch.diagonal(corrmat)
        # off_diag_up_low = (torch.triu(corrmat,diagonal=1) + torch.tril(corrmat,diagonal=-1))
        # # print('off_diag_up_low.shape',off_diag_up_low.shape)
        # # print('off_diag_up_low',off_diag_up_low)
        # off_diag = off_diag_up_low[off_diag_up_low!=0]
    
        n = corrmat.shape[0]
        mask_off = ~torch.eye(n, dtype=torch.bool)
        off_diag = corrmat[mask_off]
    
        diag_mean = torch.mean(diag).item()
        diag_median = torch.median(diag).item()
        diag_index = diag.mean() - off_diag.mean()
    
        KS_D=scipy.stats.kstest(diag.numpy(),off_diag.numpy()).statistic
        KS_pvalue=scipy.stats.kstest(diag.numpy(),off_diag.numpy()).pvalue
        
    
        self.log(f"{mode}_diag_mean", diag_mean, sync_dist=True)
        self.log(f"{mode}_diag_median", diag_median, sync_dist=True)
        self.log(f"{mode}_top1_diag_acc", top1_diag_acc, sync_dist=True)
        self.log(f"{mode}_diag_index", diag_index, sync_dist=True)
        self.log(f"{mode}_diag_rank_idx", diag_rank_idx, sync_dist=True)
        self.log(f"{mode}_KS_D", KS_D, sync_dist=True)
        self.log(f"{mode}_KS_pvalue", KS_pvalue, sync_dist=True)
        self.log(f"{mode}_mse", mse, sync_dist=True)
        self.log(f"{mode}_mae", mae, sync_dist=True)
        
        

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, batch_idx, mode="train") 
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        task = getattr(self.hparams, "downstream_task", "")

        if self.hparams.downstream_task == 'tfMRI_3D':
            subj, logits, target = self._compute_logits(batch)
            logits = logits[:,self.MNI152_mask.expand(logits.size()[1:])] # batch, valid_voxels
            target = target[:,self.MNI152_mask.expand(target.size()[1:])] # batch, valid_voxels
            output = torch.stack([logits, target], dim=2) # batch, voxels, 2
            return (subj, output.detach().cpu())
        elif task == 'rfMRI_next':
            subj, logits, target = self._compute_logits(batch)  # logits/target: [B,1,96,96,96,T]
            if self.use_fc_metrics and (self.roi_atlas_96 is not None):
                # voxel -> ROI ts -> FC vec
                roi_ts_pred = self._voxel_to_roi_ts(logits)
                roi_ts_gt   = self._voxel_to_roi_ts(target)
                feat_pred   = self._roi_ts_to_fc_vec(roi_ts_pred)   # [B,F]
                feat_gt     = self._roi_ts_to_fc_vec(roi_ts_gt)     # [B,F]
                output = torch.stack([feat_pred, feat_gt], dim=2)   # [B,F,2]
            else:
                # fallback: your old behavior
                B, _, _, _, _, T = logits.shape
                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    mask_flat = self.MNI152_mask.view(-1).bool().to(logits.device)
                    logits_f = logits.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                    target_f = target.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                else:
                    logits_f = logits.flatten(start_dim=1)
                    target_f = target.flatten(start_dim=1)
                output = torch.stack([logits_f, target_f], dim=2)
        
            if dataloader_idx == 0:
                self._rollout_log_k_curves(batch)
                self._log_rfMRI_baselines(batch, "valid")
        
            return (subj, output.detach().cpu())
            

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        outputs_valid = outputs[0]
        outputs_test = outputs[1]
        subj_valid = []
        subj_test = []
        out_valid_list = []
        out_test_list = []
        for subj, out in outputs_valid:
            subj_valid += subj
            out_valid_list.append(out)
        for subj, out in outputs_test:
            subj_test += subj
            out_test_list.append(out)
        subj_valid = np.array(subj_valid)
        subj_test = np.array(subj_test)
        total_out_valid = torch.cat(out_valid_list, dim=0)
        total_out_test = torch.cat(out_test_list, dim=0)

        # save model predictions if it is needed for future analysis
        # if ('tfMRI' in self.hparams.downstream_task) or (self.hparams.downstream_task == 'rfMRI_next'):
        #     self._save_predicted_map_and_target(subj_test, total_out_test, mode="test")
        # else:
        #     self._save_predictions(subj_valid,total_out_valid,mode="valid")
        #     self._save_predictions(subj_test,total_out_test, mode="test") 
        # (Important) Skip the sanity check stage to avoid writing to disk at the start
        
        if getattr(self.trainer, "sanity_checking", False):
            # Still run evaluation as usual, but don’t save any predictions
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
            self._evaluate_metrics(subj_test,  total_out_test,  mode="test")
            return
        
        self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
        self._evaluate_metrics(subj_test,  total_out_test,  mode="test")

        # First compute validation metrics to decide if this is a "new best"
        cur = self.trainer.callback_metrics.get(self._monitor_key)
        cur_val = float(cur) if cur is not None else None
        is_better = False
        if cur_val is not None:
            if self._best_val is None:
                is_better = True
            else:
                is_better = (cur_val > self._best_val) if (self._monitor_mode == "max") else (cur_val < self._best_val)

        is_last = (self.trainer.max_epochs is not None) and (self.current_epoch + 1 == self.trainer.max_epochs)

        if ('tfMRI' in self.hparams.downstream_task) or (self.hparams.downstream_task == 'rfMRI_next'):
            if is_better:
                self._best_val = cur_val
                self._save_predicted_map_and_target(subj_test, total_out_test, mode="test", tag="best")
            if is_last:
                self._save_predicted_map_and_target(subj_test, total_out_test, mode="test", tag="last")
        else:
            # For other tasks, prediction files are small.
            # You can keep your original saving behavior or also switch to best/last if needed
            pass

            
    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1
        
        if self.hparams.strategy == None : 
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            if (world_size > 1) and (len(self.subject_accuracy) > 0):
                total_subj_accuracy = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks     
                accuracy_dict = {}
                for dct in total_subj_accuracy:
                    for subj, metric_dict in dct.items():
                        if subj not in accuracy_dict:
                            accuracy_dict[subj] = metric_dict
                        else:
                            accuracy_dict[subj]['score']+=metric_dict['score']
                            accuracy_dict[subj]['count']+=metric_dict['count']
                self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved) 
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)


    #def _save_predicted_map_and_target(self, subj_array, total_out, mode):
        # print('total_out.device',total_out.device)
        # (total iteration/world_size) numbers of samples are passed into _evaluate_metrics.
#         subjects = np.unique(subj_array)

#         # subj_sex = []
#         subj_avg_logits = np.empty((len(subjects), total_out.shape[1])) 
#         subj_targets = np.empty((len(subjects), total_out.shape[1])) 
#         for idx, subj in enumerate(subjects):
#             #print('total_out.shape:',total_out.shape) # torch.Size([32, 132032, 2])
#             subj_logits = total_out[subj_array == subj,:,0]
#             subj_avg_logits[idx,:] = torch.mean(subj_logits, axis=0).detach().cpu().numpy() # average predicted task maps of the specific subject
#             subj_targets[idx,:] = total_out[subj_array == subj,:,1][0,:].detach().cpu().numpy()
        
#         #current_rank = torch.distributed.get_rank()
#         if self.trainer.is_global_zero:
#             os.makedirs(os.path.join('predictions',self.hparams.id),exist_ok=True)
#             with open(os.path.join('predictions',self.hparams.id,f'test_subj_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
#                 pickle.dump(subjects, pickle_out)
                
#             with open(os.path.join('predictions',self.hparams.id,f'predicted_map_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
#                 pickle.dump(subj_avg_logits, pickle_out)

#             with open(os.path.join('predictions',self.hparams.id,f'target_map_epoch{self.current_epoch}.pkl'),'wb') as pickle_out:
#                 pickle.dump(subj_targets, pickle_out)

    def _save_predicted_map_and_target(self, subj_array, total_out, mode, tag="last"):
        subjects = np.unique(subj_array)
        task = getattr(self.hparams, "downstream_task", "")

        # Aggregate to per-subject level
        subj_avg_logits = np.empty((len(subjects), total_out.shape[1]))
        subj_targets    = np.empty((len(subjects), total_out.shape[1]))
        for idx, subj in enumerate(subjects):
            pred_seg = total_out[subj_array == subj, :, 0]   # [n_seg, F]
            gt_seg   = total_out[subj_array == subj, :, 1]   # [n_seg, F]
    
            subj_avg_logits[idx, :] = pred_seg.mean(dim=0).detach().cpu().numpy()
    
            if task == "rfMRI_next":
                subj_targets[idx, :] = gt_seg.mean(dim=0).detach().cpu().numpy()   # mean
            else:
                subj_targets[idx, :] = gt_seg[0].detach().cpu().numpy()            # first (tfMRI)


        if self.trainer.is_global_zero:
            # Output directory: use PRED_OUT_DIR if available; 
            # otherwise fall back to <default_root_dir>/predictions/<run_id>
            run_id = str(getattr(self.hparams, "id", getattr(self, "run_id", "run")))
            base = os.environ.get("PRED_OUT_DIR",
                                  os.path.join(self.trainer.default_root_dir, "predictions", run_id))
            os.makedirs(base, exist_ok=True)

            # Fixed filenames (best/last), compressed with gzip; 
            # avoids generating a new file every epoch
            with gzip.open(os.path.join(base, f"test_subjects_{tag}.pkl.gz"), "wb") as f:
                pickle.dump(subjects, f, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open(os.path.join(base, f"predicted_map_{tag}.pkl.gz"), "wb") as f:
                pickle.dump(subj_avg_logits, f, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open(os.path.join(base, f"target_map_{tag}.pkl.gz"), "wb") as f:
                pickle.dump(subj_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    def test_step(self, batch, batch_idx):
        task = getattr(self.hparams, "downstream_task", "")

        # do nothing for test step since val step also performs test step
        if self.hparams.downstream_task == 'tfMRI_3D':
            subj, logits, target = self._compute_logits(batch)
            logits = logits[:,self.MNI152_mask.expand(logits.size()[1:])] # batch, valid_voxels
            target = target[:,self.MNI152_mask.expand(target.size()[1:])] # batch, valid_voxels
            output = torch.stack([logits, target], dim=2).detach().cpu() # batch, voxels, 2
            return (subj, output)
        elif self.hparams.downstream_task == 'tfMRI':
            subj, logits, target = self._compute_logits(batch)
            output = torch.stack([logits, target], dim=2) # batch, voxels, 2
            return (subj, output)
        elif task == 'rfMRI_next':
            subj, logits, target = self._compute_logits(batch)  # [B,1,96,96,96,T]
        
            if self.use_fc_metrics and (self.roi_atlas_96 is not None):
                # voxel -> ROI ts -> FC vec
                roi_ts_pred = self._voxel_to_roi_ts(logits)
                roi_ts_gt   = self._voxel_to_roi_ts(target)
                feat_pred   = self._roi_ts_to_fc_vec(roi_ts_pred)   # [B,F]
                feat_gt     = self._roi_ts_to_fc_vec(roi_ts_gt)     # [B,F]
                output      = torch.stack([feat_pred, feat_gt], dim=2)  # [B,F,2]
            else:
                # fallback: old behavior
                B, _, _, _, _, T = logits.shape
                if hasattr(self, "MNI152_mask") and self.MNI152_mask is not None:
                    mask_flat = self.MNI152_mask.view(-1).bool().to(logits.device)
                    logits_f = logits.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                    target_f = target.view(B, -1, T)[:, mask_flat, :].reshape(B, -1)
                else:
                    logits_f = logits.flatten(start_dim=1)
                    target_f = target.flatten(start_dim=1)
                output = torch.stack([logits_f, target_f], dim=2)
        
            self._log_rfMRI_baselines(batch, "test")
            return (subj, output.detach().cpu())


    def test_epoch_end(self, outputs):
        subj_test = [] 
        out_test_list = []
        for subj, out in outputs:
            subj_test += subj
            out_test_list.append(out.detach())
        subj_test = np.array(subj_test)
        total_out_test = torch.cat(out_test_list, dim=0)
        if ('tfMRI' in self.hparams.downstream_task) or (self.hparams.downstream_task == 'rfMRI_next'):
            self._save_predicted_map_and_target(subj_test, total_out_test, mode="test",tag="last")
            self._evaluate_metrics(subj_test, total_out_test, mode="test")
        else:
            self._save_predictions(subj_test, total_out_test, mode="test") 
            self._evaluate_metrics(subj_test, total_out_test, mode="test")

    def on_test_epoch_start(self) -> None:
        if self.hparams.downstream_task in ['tfMRI_3D', 'rfMRI_next']:
            self.MNI152_mask = self.MNI152_mask.to(self.device)
        return super().on_test_epoch_start()
    
    def on_train_epoch_start(self) -> None:
        if self.hparams.downstream_task in ['tfMRI_3D', 'rfMRI_next']:
            self.MNI152_mask = self.MNI152_mask.to(self.device)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings=np.zeros((self.repetitions,1))
        return super().on_train_epoch_start()
    
    def on_train_batch_start(self, batch, batch_idx):
        torch.cuda.nvtx.range_push("train") 
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx-self.gpu_warmup] = curr_time
            elif (batch_idx-self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)
                
                Throughput = (self.repetitions*self.hparams.batch_size*int(self.hparams.num_nodes) * int(self.hparams.devices))/self.total_time
                
                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:',mean_syn)
                print('std_syn:',std_syn)
                
        return super().on_train_batch_end(out, batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        torch.cuda.nvtx.range_pop() # train
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        if self.hparams.downstream_task in ['tfMRI_3D', 'rfMRI_next']:
            self.MNI152_mask = self.MNI152_mask.to(self.device)
        torch.cuda.nvtx.range_push("valid")
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        torch.cuda.nvtx.range_pop()
        return super().on_validation_epoch_end()

    def on_before_backward(self, loss: torch.Tensor) -> None:
        torch.cuda.nvtx.range_push("backward")
        return super().on_before_backward(loss)

    def on_after_backward(self) -> None:
        torch.cuda.nvtx.range_pop()
        return super().on_after_backward()

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")
        
        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            warmup = int(total_iterations * self.hparams.warmup) #?
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * self.hparams.warmup) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 2 #? 1 in SwiFUN
            
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        ## training related
        # group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        # group.set_defaults(grad_clip=False)
        group.add_argument("--id", type=str, default=None,
                   help="run identifier used to name prediction output folder")

        group.add_argument("--loss_type", type=str, default="mse", help="which loss to use. You can use reconstructive-contrastive loss with 'rc'")
        group.add_argument("--within_subj_margin", type=float, default=0.34,
                   help="lambda in RC loss (weight for reconstruction term)")
        group.add_argument("--across_subj_margin", type=float, default=0.66,
                   help="(1-lambda) in RC loss (weight for contrast term)")
        
        
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.set_defaults(use_scheduler=False)
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--warmup", type=float, default=0.01, help="warmup in CosineAnnealingWarmUpRestarts (recommend 0.01~0.1 values)")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=0.5, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")
        
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.set_defaults(augment_during_training=False)
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        
        ## model related
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--out_chans", type=int, default=1, help="Channel size of target output")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        # group.add_argument("--window_size",  type=int, default=7, help="window size from the second layers")
        # group.add_argument("--patch_size",  type=int, default=2, help="patch size")
        group.add_argument("--window_size", nargs="+", type=int, default=[7], help="window size (D H W [T])")
        group.add_argument("--first_window_size", nargs="+", type=int, default=None, help="first-stage window size (D H W [T])")
        group.add_argument("--patch_size", nargs="+", type=int, default=[2], help="patch size (D H W [T])")

        group.add_argument("--use_v2", action='store_true', help="whether to use SwinUNETR v2")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage of encoder")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")

        ## others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")    
        
        #rIMRTI
        group.add_argument("--pred_context", type=int, default=None,
                   help="Number of context frames; None = use the first half")
        group.add_argument("--pred_horizon", type=int, default=None,
                           help="Number of future frames to predict at once; None = use the second half; strict next-token prediction = 1")
        group.add_argument("--k_max", type=int, default=5,
                           help="Maximum number of autoregressive rollout steps during validation (k=1..k_max)")
        group.add_argument("--teacher_forcing_ratio", type=float, default=0.0,
                   help="probability to use ground-truth frame at each AR step")
        group.add_argument("--tf_schedule", type=str, default="linear", choices=["constant","linear","cosine","exp"])

        group.add_argument("--ctx_update", type=str, default="sliding",
                   choices=["sliding", "growing"],
                   help="how to update ctx during rfMRI_next rollout")

        #residual learning (rfMRI_next)
        group.add_argument("--use_residual_branch", action="store_true",
                           help="If set, model predicts residual over a baseline (persist/ctxmean) for rfMRI_next")
        group.set_defaults(use_residual_branch=False)
        
        group.add_argument("--residual_base", type=str, default="persist",
                           choices=["persist", "ctxmean"],
                           help="baseline type for residual learning")
        
        group.add_argument("--residual_l2", type=float, default=0.0,
                           help="optional L2 penalty on residual (0 disables)")

        #metric
        group.add_argument("--use_fc_metrics", action="store_true",
                   help="If set, use ROI-FC vector for rfMRI_next corrmat/top1/KS metrics")
        group.set_defaults(use_fc_metrics=False)
        
        group.add_argument("--roi_atlas_pt", type=str, default=None,
                   help="Path to ROI atlas tensor saved by torch.save, shape [96,96,96], int labels (0 background)")


        return parser
