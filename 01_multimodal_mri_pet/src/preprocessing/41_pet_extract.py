"""
41_pet_extract.py
-----------------
Tracer-specific PET quantification in each subject's own MRI space.

For every subject with a FastSurfer segmentation and a SCAN-standardised PET series of the
requested kind (TAU or AMY):
  1. pick ONE series: "Coreg, Avg, Rigid Reg to Std Img/Vox Size, <window>, 6mm Res"
     (uniform 6 mm resolution), preferred uptake window, date closest to the MRI date
  2. DICOM -> NIfTI with dcm2niix (keeps the scanner geometry)
  3. rigid registration PET -> T1 (FastSurfer orig.mgz, brain-masked) with ANTs, mutual information
  4. mean uptake and voxel count for every FastSurfer label, plus reference-region means
  5. QC numbers and an overlay PNG

No label information is used. Output per subject: ~/alz/p1/work/pet_feat/<kind>/<NACCID>.json
Resume-friendly: finished subjects are skipped.

Usage: python 41_pet_extract.py TAU      (or AMY)
"""
import os, sys, re, json, glob, shutil, subprocess, tempfile, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")

ALZ = Path.home() / "alz"; P1 = ALZ / "p1"; FS = ALZ / "fs_output"; NII = ALZ / "nii_all"
KIND = sys.argv[1] if len(sys.argv) > 1 else "TAU"
OUT = P1 / "work" / "pet_feat" / KIND; OUT.mkdir(parents=True, exist_ok=True)
REG = P1 / "work" / "pet_reg" / KIND; REG.mkdir(parents=True, exist_ok=True)
QC = P1 / "work" / "pet_qc" / KIND; QC.mkdir(parents=True, exist_ok=True)

KIND_OF = {"T80": "TAU", "M62": "TAU", "P26": "TAU", "PIB": "AMY", "FBB": "AMY", "AV": "AMY", "NAV": "AMY"}
TRACER = {"T80": "flortaucipir", "M62": "MK-6240", "P26": "PI-2620", "PIB": "PiB", "FBB": "florbetaben",
          "AV": "florbetapir", "NAV": "NAV4694"}
# preferred uptake windows (minutes post injection), best first
WINDOWS = {"T80": ["80-100"], "M62": ["90-110", "70-110", "70-90"], "P26": ["45-75", "30-60"],
           "PIB": ["50-70", "40-70", "40-60"], "FBB": ["90-110"], "AV": ["50-70"], "NAV": ["50-70", "40-70"]}
STD = "Coreg__Avg__Rigid_Reg_to_Std_Img_Vox_Size"


def parse_date(s):
    return pd.to_datetime(s[:10], errors="coerce")


def mri_date(sid, mri_inv):
    """Date of the T1 that FastSurfer used: match the dcm2niix sidecar to the MRI inventory."""
    try:
        js = json.load(open(NII / f"{sid}.json"))
    except Exception:
        return None, "no_sidecar"
    rows = mri_inv[mri_inv.NACCID == sid]
    want = str(js.get("SeriesDescription", "")).replace(",", ";")
    m = rows[rows.SeriesDescription == want]
    if len(m) == 0:
        m = rows[rows.series_dir.str.lower().str.contains("mprage|t1|spgr")]
        how = "name_fallback"
    else:
        how = "series_description"
    dates = sorted(set(m.date_dir))
    if not dates:
        return None, "no_match"
    if len(dates) == 1:
        return parse_date(dates[0]), how
    # several sessions: compare the acquisition time of day
    t = str(js.get("AcquisitionTime", "")).replace(":", "")[:6]
    for d in dates:
        if d[11:19].replace("_", "")[:4] == t[:4]:
            return parse_date(d), how + "+time"
    return parse_date(dates[-1]), how + "+latest"


def pick_series(rows, mdate):
    """rows: standardised 6 mm series of one kind for one subject."""
    best = None
    for _, r in rows.iterrows():
        pre = r.series_dir.split("_")[0]
        w = re.search(r"Size__(.*?)__6mm", r.series_dir)
        w = w.group(1) if w else ""
        base = re.sub(r"[AB]+$", "", w)
        rank_w = WINDOWS[pre].index(base) if base in WINDOWS[pre] else 9
        rep = 0 if w == base else 1
        d = parse_date(r.date_dir)
        gap = abs((d - mdate).days) if (mdate is not None and pd.notna(d)) else 99999
        key = (gap, rank_w, rep, r.date_dir)
        if best is None or key < best[0]:
            best = (key, r, pre, w, gap)
    return best


def overlay_png(t1, pet, seg, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = [int(x) for x in ndimage.center_of_mass(seg > 0)]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        for i, a in enumerate(ax):
            sl = [slice(None)] * 3; sl[i] = c[i]
            a.imshow(np.rot90(t1[tuple(sl)]), cmap="gray")
            a.imshow(np.rot90(pet[tuple(sl)]), cmap="hot", alpha=0.45, vmin=0, vmax=np.percentile(pet[seg > 0], 99.5))
            a.contour(np.rot90((seg[tuple(sl)] > 0).astype(float)), levels=[0.5], colors="cyan", linewidths=0.5)
            a.axis("off")
        fig.tight_layout(pad=0.1); fig.savefig(path, dpi=70); plt.close(fig)
    except Exception:
        pass


def process(job):
    sid, row, pre, window, gap, mdate, mhow = job
    out = OUT / f"{sid}.json"
    if out.exists():
        return sid, "skip"
    tmp = tempfile.mkdtemp(prefix=f"pet_{sid}_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        import ants
        t1_img = nib.load(str(FS / sid / "mri" / "orig.mgz"))
        seg_img = nib.load(str(FS / sid / "mri" / "aparc.DKTatlas+aseg.deep.mgz"))
        mask_img = nib.load(str(FS / sid / "mri" / "mask.mgz"))
        t1 = np.asarray(t1_img.dataobj).astype(np.float32)
        seg = np.asarray(seg_img.dataobj).astype(np.int32)
        mask = np.asarray(mask_img.dataobj) > 0
        aff = t1_img.affine
        fixed_f = os.path.join(tmp, "t1brain.nii.gz")
        nib.save(nib.Nifti1Image(t1 * mask, aff), fixed_f)

        r = subprocess.run(["dcm2niix", "-z", "y", "-b", "n", "-o", tmp, "-f", "pet", row.path],
                           capture_output=True, text=True, timeout=300)
        cand = sorted(glob.glob(os.path.join(tmp, "pet*.nii.gz")), key=os.path.getsize, reverse=True)
        if not cand:
            return sid, "dcm2niix_fail:" + r.stdout[-120:].replace("\n", " ")
        p_img = nib.load(cand[0]); p = np.asarray(p_img.dataobj).astype(np.float32)
        frames = 1
        if p.ndim == 4:
            frames = p.shape[3]; p = p.mean(3)
        p[~np.isfinite(p)] = 0; p[p < 0] = 0
        moving_f = os.path.join(tmp, "pet3d.nii.gz")
        nib.save(nib.Nifti1Image(p, p_img.affine), moving_f)

        fixed = ants.image_read(fixed_f); moving = ants.image_read(moving_f)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid", aff_metric="mattes",
                                aff_sampling=32, random_seed=1, verbose=False)
        warped_f = str(REG / f"{sid}.nii.gz")
        ants.image_write(reg["warpedmovout"], warped_f)
        pet = np.asarray(nib.load(warped_f).dataobj).astype(np.float32)
        if pet.shape != seg.shape:
            return sid, f"shape_mismatch:{pet.shape}"

        # ---- region statistics
        labels = np.unique(seg); labels = labels[labels > 0]
        sums = ndimage.sum(pet, seg, labels); cnt = ndimage.sum(np.ones_like(pet), seg, labels)
        regions = {int(l): [float(s / c), int(c)] for l, s, c in zip(labels, sums, cnt) if c > 0}

        # ---- reference regions
        ijk = np.argwhere(np.isin(seg, [8, 47]))
        z = (aff @ np.c_[ijk, np.ones(len(ijk))].T)[2]            # world superior-inferior coordinate (mm)
        cer_gm = np.isin(seg, [8, 47])
        inf = np.zeros_like(cer_gm); sel = ijk[z < np.median(z)]
        inf[sel[:, 0], sel[:, 1], sel[:, 2]] = True
        inf_er = ndimage.binary_erosion(inf, iterations=1)
        whole = np.isin(seg, [7, 8, 46, 47])
        wm = ndimage.binary_erosion(np.isin(seg, [2, 41]), iterations=2)
        refs = {"cereb_gm": float(pet[cer_gm].mean()), "cereb_gm_inferior": float(pet[inf].mean()),
                "cereb_gm_inferior_eroded": float(pet[inf_er].mean()) if inf_er.sum() > 100 else float("nan"),
                "cereb_whole": float(pet[whole].mean()), "wm_eroded": float(pet[wm].mean()) if wm.sum() > 100 else float("nan"),
                "n_cereb_gm": int(cer_gm.sum()), "n_cereb_gm_inferior_eroded": int(inf_er.sum())}

        # ---- QC
        thr = 0.3 * np.percentile(pet[pet > 0], 99) if (pet > 0).any() else 0
        head = pet > thr
        brain = seg > 0
        dice = 2 * (head & brain).sum() / (head.sum() + brain.sum() + 1e-6)
        csf = np.isin(seg, [4, 43]); gmctx = seg >= 1000
        qc = {"dice_pet_brain": float(dice),
              "brain_in_fov": float((pet[brain] > 0).mean()),
              "csf_over_cortex": float(pet[csf].mean() / (pet[gmctx].mean() + 1e-6)) if csf.sum() > 50 else float("nan"),
              "inside_over_outside": float(pet[brain].mean() / (pet[~ndimage.binary_dilation(brain, iterations=8)].mean() + 1e-6)),
              "frames": int(frames), "pet_shape": list(p.shape), "pet_zooms": [float(x) for x in p_img.header.get_zooms()[:3]]}
        overlay_png(t1, pet, seg, str(QC / f"{sid}.png"))

        json.dump({"NACCID": sid, "kind": KIND, "prefix": pre, "tracer": TRACER[pre], "window": window,
                   "series_dir": row.series_dir, "pet_date": row.date_dir[:10], "image_id": row.image_id,
                   "radiopharmaceutical_tag": row.Radiopharmaceutical, "manufacturer": row.Manufacturer,
                   "model": row.ManufacturerModelName,
                   "mri_date": str(mdate.date()) if mdate is not None and pd.notna(mdate) else "", "mri_date_method": mhow,
                   "pet_mri_gap_days": int(gap) if gap < 99999 else None,
                   "refs": refs, "qc": qc, "regions": regions}, open(out, "w"))
        return sid, "done"
    except Exception as e:
        return sid, "err:" + type(e).__name__ + ":" + str(e)[:150] + " | " + traceback.format_exc().splitlines()[-2][:120]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    inv = pd.read_csv(P1 / "work" / "pet_inventory.csv", dtype=str).fillna("")
    mri_inv = pd.read_csv(P1 / "work" / "mri_inventory.csv", dtype=str).fillna("")
    inv["pre"] = inv.series_dir.str.split("_").str[0]
    std = inv[inv.series_dir.str.contains(STD) & inv.series_dir.str.contains("6mm_Res") & (inv.pre.map(KIND_OF) == KIND)]
    fs_done = {s for s in os.listdir(FS) if (FS / s / "stats" / "aseg+DKT.VINN.stats").exists()}
    subs = sorted(set(std.NACCID) & fs_done)
    lo = int(os.environ.get("START", 0)); hi = int(os.environ.get("END", len(subs)))
    print(f"{KIND}: {len(subs)} subjects with standardised series + FastSurfer; running [{lo}:{hi}]", flush=True)
    jobs = []
    for sid in subs[lo:hi]:
        md, how = mri_date(sid, mri_inv)
        b = pick_series(std[std.NACCID == sid], md)
        jobs.append((sid, b[1], b[2], b[3], b[4], md, how))
    nw = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) // 2)
    stat = {}
    with Pool(nw) as pool:
        for i, (sid, st) in enumerate(pool.imap_unordered(process, jobs)):
            stat[st.split(":")[0]] = stat.get(st.split(":")[0], 0) + 1
            if not st.startswith(("done", "skip")):
                print(f"  {sid} {st}", flush=True)
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(jobs)}] {stat}", flush=True)
    print(f"FINAL {KIND}: {stat}  json files: {len(list(OUT.glob('*.json')))}", flush=True)


if __name__ == "__main__":
    main()
