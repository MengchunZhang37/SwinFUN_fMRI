#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert fMRIPrep outputs (MNI + brain mask) to SwiFUN-ready directory:
- crop non-brain background with a tight bbox from mask to ensure each axis <= 96
- whole-brain normalization (z-norm or min-max) on non-background voxels
- save each TR as FP16 torch .pt  (frame_0.pt ... frame_{T-1}.pt)
- save global_stats.pt (valid_voxels, mean, std, max)
- optionally create metadata/metafile.csv template

This follows the official SwiFUN guidance:
- crop background to keep dims <= 96; pad to 96³ later in Dataset class
- whole-brain z-normalization (main) or min-max
- split 4D fMRI into per-TR .pt files (FP16), store global stats
- directory structure: {DATASET}_MNI_to_TRs/{img/sub-XX/*.pt, metadata/metafile.csv}
Refs: README + preprocessing examples. 
"""

import os, glob, csv
import numpy as np
import nibabel as nib
import torch

# ========= USER CONFIG =========
FMRIPREP_ROOT = "/ix1/haizenstein/liw82/ds000030_out"  # fMRIPrep derivatives root
DATASET_NAME  = "DS000030"                             # just a name tag
SAVE_ROOT     = f"/ihome/haizenstein/mez141/ondemand/{DATASET_NAME}_cleaned_data"  # output root
SCALING       = "z-norm"  # choose: "z-norm" or "minmax"
MAKE_METADATA = True      # set False if you already have metafile.csv
SUBJECT_GLOB  = "sub-*"   # which subjects to include
TASK_GLOB     = "*task-rest*"  # which runs; e.g. rest only. Change if needed.
# =================================

IMG_DIR   = os.path.join(SAVE_ROOT, "img")
META_DIR  = os.path.join(SAVE_ROOT, "metadata")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def find_runs(fmriprep_root, sub_glob="sub-*", task_glob="*task-rest*"):
    """
    Traverse each subject directory and recursively search under func/,
    collect REST runs with MNI+preproc + corresponding brain_mask.
    Compatible hierarchies: sub-XXXX/func/... and sub-XXXX/sub-XXXX/func/...
    Return: [(sub_name, bold_file, mask_file), ...]
    """
    sub_dirs = sorted([d for d in glob.glob(os.path.join(fmriprep_root, sub_glob))
                       if os.path.isdir(d)])
    pairs = []

    for sdir in sub_dirs:
        sub = os.path.basename(sdir)  # always use the outermost subject name
        # First check the standard location: sub-XXX/func; if not found, recursively search for any func/ under this subject directory
        func_dirs = []
        std_func = os.path.join(sdir, "func")
        if os.path.isdir(std_func):
            func_dirs.append(std_func)
        else:
            # Recursively search for func directories at any depth
            func_dirs = sorted(glob.glob(os.path.join(sdir, "**", "func"), recursive=True))

        if not func_dirs:
            # This subject has no func, skip
            continue

        # To be safe: merge all func directories (some subjects may have multiple sessions or nested structures)
        seen = set()
        for fdir in func_dirs:
            if fdir in seen:
                continue
            seen.add(fdir)

            # Priority: MNI space + desc-preproc
            pat = os.path.join(fdir, f"{task_glob}*space-MNI152NLin2009cAsym_*desc-preproc_bold.nii.gz")
            bolds = sorted(glob.glob(pat))
            if not bolds:
                # Fallback: rare cases without space- naming
                pat_fb = os.path.join(fdir, f"{task_glob}*desc-preproc_bold.nii.gz")
                bolds = sorted(glob.glob(pat_fb))

            for bf in bolds:
                mf = bf.replace("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz")
                if os.path.exists(mf):
                    pairs.append((sub, bf, mf))
                else:
                    print(f"[WARN] {sub}: missing brain_mask -> skip {os.path.basename(bf)}")

    return pairs


def tight_bbox_from_mask(mask3d: np.ndarray, maxlen=96):
    """
    Get tight bounding box from mask, then center-crop each axis to <= maxlen.
    """
    idx = np.where(mask3d > 0)
    if idx[0].size == 0:
        # empty mask; fallback to whole volume (will be clipped later)
        x0, x1 = 0, mask3d.shape[0]
        y0, y1 = 0, mask3d.shape[1]
        z0, z1 = 0, mask3d.shape[2]
    else:
        x0, x1 = idx[0].min(), idx[0].max() + 1
        y0, y1 = idx[1].min(), idx[1].max() + 1
        z0, z1 = idx[2].min(), idx[2].max() + 1

    def clip_to_96(a0, a1):
        length = a1 - a0
        if length <= maxlen:
            return a0, a1
        # center-crop to maxlen
        extra = length - maxlen
        shift = extra // 2
        return a0 + shift, a1 - (extra - shift)

    x0, x1 = clip_to_96(x0, x1)
    y0, y1 = clip_to_96(y0, y1)
    z0, z1 = clip_to_96(z0, z1)
    return (slice(x0, x1), slice(y0, y1), slice(z0, z1))

def normalize_whole_brain(data4d: np.ndarray, mask3d: np.ndarray, mode="z-norm"):
    """
    Apply whole-brain normalization ONLY on non-background voxels defined by mask.
    Background will be filled with 0 or the min of normalized data (min-max case ~0).
    """
    bg = (mask3d == 0)
    brain = ~bg
    # reshape mask to 4D for broadcasting over time
    brain4d = np.broadcast_to(brain[..., None], data4d.shape)
    bg4d    = ~brain4d

    data = data4d.astype(np.float32, copy=False)
    vox = data[brain4d]

    if vox.size == 0:
        # fallback: nothing to normalize
        return np.zeros_like(data, dtype=np.float32), 0.0, 1.0, 0.0, 0

    if mode == "z-norm":
        mu  = float(vox.mean())
        std = float(vox.std() + 1e-6)
        temp = (data - mu) / std
        fill0 = float(temp[brain4d].min())  # could be negative
    elif mode == "minmax":
        vmin = float(vox.min())
        vmax = float(vox.max())
        temp = (data - vmin) / (vmax - vmin + 1e-6)
        fill0 = 0.0
        mu, std = float(vox.mean()), float(vox.std())  # record stats on raw vox
    else:
        raise ValueError("SCALING must be 'z-norm' or 'minmax'.")

    out = np.empty_like(temp, dtype=np.float32)
    out[brain4d] = temp[brain4d]
    out[bg4d]    = fill0

    valid_voxels = int(brain.sum())
    global_max   = float(data[brain].max())
    return out, mu, std, global_max, valid_voxels

def save_subject_frames(sub: str, data4d_norm: np.ndarray, save_root: str,
                        mu: float, std: float, gmax: float, valid_voxels: int):
    """
    Save per-TR frames as FP16 .pt and global_stats.pt
    """
    sdir = os.path.join(save_root, "img", sub)
    os.makedirs(sdir, exist_ok=True)
    T = data4d_norm.shape[3]
    tens = torch.from_numpy(data4d_norm.astype(np.float16, copy=False))  # (X,Y,Z,T)
    for t in range(T):
        # keep last dim as singleton channel, like examples
        vol = torch.from_numpy(data4d_norm[..., t].astype(np.float16, copy=False))  # (H,W,D)
        torch.save(vol, os.path.join(sdir, f"frame_{t}.pt"))

    stats = {
        "valid_voxels": valid_voxels,
        "global_mean":  mu,
        "global_std":   std,
        "global_max":   gmax,
    }
    torch.save(stats, os.path.join(sdir, "global_stats.pt"))

def process_one(sub: str, bold_file: str, mask_file: str):
    """
    Load MNI preproc bold + MNI brain mask from fMRIPrep, crop to <=96³,
    whole-brain normalize, then write FP16 .pt files + stats.
    """
    print(f"[proc] {sub} | bold={os.path.basename(bold_file)}")
    img  = nib.load(bold_file)
    mask = nib.load(mask_file)
    data4d = img.get_fdata(dtype=np.float32)  # (X,Y,Z,T)
    mask3d = (mask.get_fdata() > 0).astype(np.uint8)

    # tight bbox -> center-crop to <=96 per axis
    sx, sy, sz = tight_bbox_from_mask(mask3d, maxlen=96)
    data4d_c   = data4d[sx, sy, sz, :]
    mask3d_c   = mask3d[sx, sy, sz]

    # normalization (z-norm or minmax) on brain voxels
    data4d_n, mu, std, gmax, valid_voxels = normalize_whole_brain(
        data4d_c, mask3d_c, mode=SCALING
    )

    save_subject_frames(sub, data4d_n, SAVE_ROOT, mu, std, gmax, valid_voxels)

# def maybe_write_metafile(pairs):
#     """
#     Create a minimal metadata/metafile.csv template with subject_name and a stub column
#     for your supervision (e.g., task z-map path or label).
#     """
#     meta_path = os.path.join(META_DIR, "metafile.csv")
#     if not MAKE_METADATA:
#         return
#     rows = []
#     for sub, _, _ in pairs:
#         rows.append({
#             "subject_name": sub,
#             "target": ""  # TODO: fill with your task z-map path or class label
#         })
#     with open(meta_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["subject_name", "target"])
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"[meta] wrote template: {meta_path}")

def main():
    pairs = find_runs(FMRIPREP_ROOT, SUBJECT_GLOB, TASK_GLOB)
    if not pairs:
        print("[ERR] No fMRIPrep MNI runs found. Check paths/globs.")
        return
    print(f"[info] found {len(pairs)} runs")

    for sub, bold, mask in pairs:
        process_one(sub, bold, mask)

    # maybe_write_metafile(pairs)
    print(f"[done] Output at: {SAVE_ROOT}")
    print("Structure:")
    print(f"{os.path.basename(SAVE_ROOT)}/")
    print("  img/sub-XXXX/frame_0.pt ... frame_T.pt + global_stats.pt")
    print("  metadata/metafile.csv (template)")

if __name__ == "__main__":
    main()
