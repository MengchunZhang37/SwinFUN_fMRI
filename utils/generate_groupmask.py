#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

# ================================================
FMRIPREP_ROOT = "/ix1/haizenstein/liw82/ds000030_out"   # fMRIPrep derivatives root
REF_BOLD = "/ix1/haizenstein/liw82/ds000030_out/sub-10159/func/sub-10159_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
OUT_NII  = "/ihome/haizenstein/mez141/ondemand/DS000030_cleaned_data/ROI/group_p0.8_96.nii.gz"
OUT_PT   = "/ihome/haizenstein/mez141/ondemand/DS000030_cleaned_data/ROI/group_p0.8_96.pt"
P_THRESH = 0.8  # consensus threshold p
# ========= Fixed cropping window (must match your model input) =========
# Example: center-crop to ≤96 (you can also fill in explicit indices)
FIXED_CROP = None   # in the form (x0,x1,y0,y1,z0,z1); None means use overall tight bbox then center-crop to ≤96
# ================================================

def list_mni_masks(root):
    # Collect all brain_mask files in MNI space (recursive, supports sub-X/sub-X/func)
    pat = str(Path(root) / "sub-*" / "**" / "func" / "*space-MNI152NLin2009cAsym_*desc-brain_mask.nii.gz")
    return sorted(glob.glob(pat, recursive=True))

def load_ref0(ref_bold):
    img = nib.load(ref_bold)
    # Use the first frame as 3D reference, preserve 3×3×4mm grid
    ref0 = nib.Nifti1Image(img.get_fdata()[...,0], img.affine, img.header)
    return ref0

def resample_nn(src_img, ref_img):
    # Nearest-neighbor "manual resampling" (avoid extra library dependency)
    # Approach: project ref voxel centers to src coords via ref affine, then sample
    # Here we use nilearn if available, otherwise fall back to nibabel grid with NN
    try:
        from nilearn.image import resample_to_img
        return resample_to_img(src_img, ref_img, interpolation="nearest")
    except Exception:
        import numpy.linalg as npl
        ref_shape = ref_img.shape
        ref_aff   = ref_img.affine
        src_data  = src_img.get_fdata()
        src_aff   = src_img.affine
        inv_src_aff = npl.inv(src_aff)
        xs, ys, zs = [np.arange(s) for s in ref_shape]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        ref_idx = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)  # (...,4)
        ref_xyz = ref_idx @ ref_aff.T
        src_ijk = (ref_xyz @ inv_src_aff.T)[...,:3]
        src_ijk_round = np.rint(src_ijk).astype(int)
        for d in range(3):
            src_ijk_round[..., d] = np.clip(src_ijk_round[..., d], 0, src_data.shape[d]-1)
        out = src_data[tuple(np.moveaxis(src_ijk_round, -1, 0))]
        return nib.Nifti1Image(out, ref_aff, ref_img.header)

def union_per_subject(mask_paths_by_subj, ref0):
    """Union multiple runs for the same subject"""
    subj_union = {}
    for sub, paths in mask_paths_by_subj.items():
        acc = None
        for p in paths:
            m = nib.load(p)
            m_rs = resample_nn(m, ref0).get_fdata() > 0.5
            acc = m_rs if acc is None else (acc | m_rs)
        subj_union[sub] = acc.astype(np.uint8)
    return subj_union

def crop_to_96(vol, crop=None):
    """Crop to ≤96 and pad to (96,96,96)"""
    x0=x1=y0=y1=z0=z1=None
    if crop is None:
        idx = np.where(vol > 0)
        if idx[0].size == 0:
            # If empty, just center window
            shape = vol.shape
            x0,x1 = (shape[0]-96)//2, (shape[0]-96)//2 + min(96,shape[0])
            y0,y1 = (shape[1]-96)//2, (shape[1]-96)//2 + min(96,shape[1])
            z0,z1 = (shape[2]-96)//2, (shape[2]-96)//2 + min(96,shape[2])
        else:
            x0,x1 = idx[0].min(), idx[0].max()+1
            y0,y1 = idx[1].min(), idx[1].max()+1
            z0,z1 = idx[2].min(), idx[2].max()+1
            def clip(a0,a1,maxlen=96):
                L=a1-a0
                if L<=maxlen: return a0,a1
                s=(L-maxlen)//2; return a0+s, a1-(L-maxlen-s)
            x0,x1 = clip(x0,x1); y0,y1 = clip(y0,y1); z0,z1 = clip(z0,z1)
    else:
        x0,x1,y0,y1,z0,z1 = crop

    cropped = vol[x0:x1, y0:y1, z0:z1]
    pad = lambda L: (0, max(0, 96-L))
    px, py, pz = pad(cropped.shape[0]), pad(cropped.shape[1]), pad(cropped.shape[2])
    out = np.pad(cropped, (px,py,pz), mode="constant", constant_values=0)
    return out[:96,:96,:96], (x0,x1,y0,y1,z0,z1)

def main():
    ref0 = load_ref0(REF_BOLD)
    # Collect all masks and group by subject
    all_masks = list_mni_masks(FMRIPREP_ROOT)
    if not all_masks:
        raise RuntimeError("No MNI brain_mask found (check path/permissions)")

    by_sub = {}
    for p in all_masks:
        # Relative to fmriprep root, grab the first sub- segment as subject name
        rel = os.path.relpath(p, FMRIPREP_ROOT).split(os.sep)
        sub = next(s for s in rel if s.startswith("sub-"))
        by_sub.setdefault(sub, []).append(p)

    print(f"subjects with masks: {len(by_sub)}")

    subj_union = union_per_subject(by_sub, ref0)
    stack = np.stack([subj_union[k] for k in sorted(subj_union.keys())], axis=0)  # (N, X, Y, Z)
    meanmap = stack.mean(axis=0)

    group = (meanmap >= P_THRESH).astype(np.uint8)

    # Crop to ≤96 and pad to 96³ (fixed window can be passed via FIXED_CROP)
    group96, used_crop = crop_to_96(group, crop=FIXED_CROP)
    print("Used crop:", used_crop, " -> out shape:", group96.shape)

    # Save NIfTI (use ref0’s affine)
    nib.save(nib.Nifti1Image(group96.astype(np.uint8), ref0.affine, ref0.header), OUT_NII)
    print("Saved NIfTI:", OUT_NII)

    # Save torch .pt (float32 or bool both okay)
    torch.save(torch.from_numpy(group96.astype(np.float32)), OUT_PT)
    print("Saved PT:", OUT_PT)

if __name__ == "__main__":
    main()
