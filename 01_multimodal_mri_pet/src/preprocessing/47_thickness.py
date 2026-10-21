"""
47_thickness.py
---------------
Regional cortical thickness from our own T1 processing (no FreeSurfer surfaces, no labels used).

Per subject:
  1. FastSurfer segmentation (aparc.DKTatlas+aseg.deep.mgz, conformed 1 mm) -> cortical grey matter (DKT labels >= 1000)
     and an "inner" class (cerebral white matter, deep grey nuclei, lateral ventricles, choroid plexus, hippocampus, amygdala)
  2. crop to the cerebrum bounding box, lightly smoothed class maps as probability images
  3. ANTs DiReCT (KellyKapowski) voxel-wise cortical thickness
  4. mean thickness of the voxels of every DKT parcel

Output: ~/alz/p1/work/thickness/<NACCID>.json ; combined table ~/alz/p1/work/features_thickness.csv (prefix thk_)
"""
import os, sys, json, time, tempfile, shutil
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage

THREADS = int(os.environ.get("THK_THREADS", 3))
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(THREADS)
ALZ = Path.home() / "alz"; P1 = ALZ / "p1"; FS = ALZ / "fs_output"
OUT = P1 / "work" / "thickness"; OUT.mkdir(parents=True, exist_ok=True)
INNER = [2, 41, 77, 10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60, 4, 43, 31, 63, 17, 53, 18, 54]


def lut(sid):
    names = {}
    for line in open(FS / sid / "stats" / "aseg+DKT.VINN.stats"):
        p = line.split()
        if line.startswith("#") or len(p) < 6:
            continue
        try:
            names[int(p[1])] = p[4].replace("ctx-lh-", "L-").replace("ctx-rh-", "R-")
        except ValueError:
            pass
    return names


def process(sid):
    out = OUT / f"{sid}.json"
    if out.exists():
        return sid, "skip", 0.0
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix=f"thk_{sid}_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        import ants
        img = nib.load(str(FS / sid / "mri" / "aparc.DKTatlas+aseg.deep.mgz"))
        seg = np.asarray(img.dataobj).astype(np.int32); vox = [float(z) for z in img.header.get_zooms()[:3]]
        gm = seg >= 1000; wm = np.isin(seg, INNER)
        idx = np.argwhere(gm | wm); lo = np.maximum(idx.min(0) - 6, 0); hi = np.minimum(idx.max(0) + 7, seg.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        segc, gmc, wmc = seg[sl], gm[sl], wm[sl]
        s3 = np.zeros(segc.shape, np.float32); s3[gmc] = 2; s3[wmc] = 3
        gp = ndimage.gaussian_filter(gmc.astype(np.float32), 0.5); wp = ndimage.gaussian_filter(wmc.astype(np.float32), 0.5)
        mk = lambda a: ants.from_numpy(a.astype(np.float32), spacing=vox)
        th = ants.kelly_kapowski(s=mk(s3), g=mk(gp), w=mk(wp), its=45, r=0.025, m=1.5, x=0, verbose=0).numpy()
        labels = np.unique(segc[gmc]); names = lut(sid)
        valid = gmc & (th > 0.1)
        res = {}
        for l in labels:
            v = th[valid & (segc == l)]
            if v.size >= 30:
                res[names.get(int(l), str(int(l)))] = [float(v.mean()), float(np.median(v)), int(v.size)]
        allv = th[valid]
        json.dump(dict(NACCID=sid, voxel_mm=vox, n_gm_vox=int(gmc.sum()), frac_valid=float(valid.sum() / max(gmc.sum(), 1)),
                       mean_thickness=float(allv.mean()), regions=res, seconds=time.time() - t0, threads=THREADS,
                       params=dict(its=45, r=0.025, m=1.5, prior_mm=10, smooth_sigma_vox=0.5)), open(out, "w"))
        return sid, "done", time.time() - t0
    except Exception as e:
        return sid, "err:" + type(e).__name__ + ":" + str(e)[:150], time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collect():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        j = json.load(open(f)); r = {"NACCID": j["NACCID"], "has_thk": 1, "thk_mean_cortex": j["mean_thickness"]}
        reg = {k: v[0] for k, v in j["regions"].items()}
        r.update({f"thk_{k}": v for k, v in reg.items()})
        for k in sorted({k[2:] for k in reg}):
            l, rr = reg.get("L-" + k), reg.get("R-" + k)
            if l is not None and rr is not None:
                r[f"thk_bi_{k}"] = (l + rr) / 2; r[f"thk_asym_{k}"] = (l - rr) / (l + rr)
        sig = [reg.get(h + k) for h in ("L-", "R-") for k in ("entorhinal", "inferiortemporal", "middletemporal", "fusiform")]
        sig = [x for x in sig if x is not None]
        if sig:
            r["thk_ad_signature"] = float(np.mean(sig))      # temporal meta-ROI regions
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(P1 / "work" / "features_thickness.csv", index=False)
    print(f"collected {len(df)} subjects, {df.shape[1] - 2} thickness features; mean cortical thickness "
          f"{df.thk_mean_cortex.mean():.2f} +- {df.thk_mean_cortex.std():.2f} mm", flush=True)


def main():
    lab = pd.read_csv(ALZ / "data" / "subset450_labels.csv")
    subs = sorted(s for s in lab.NACCID if (FS / s / "stats" / "aseg+DKT.VINN.stats").exists())
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        subs = subs[:int(sys.argv[2])]
    if len(sys.argv) > 1 and sys.argv[1] == "shard":                  # shard k n: every n-th subject starting at k
        subs = subs[int(sys.argv[2])::int(sys.argv[3])]
    todo = [s for s in subs if not (OUT / f"{s}.json").exists()]
    nw = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) // THREADS)
    print(f"{len(subs)} subjects, {len(todo)} to do, {nw} workers x {THREADS} threads", flush=True)
    stat = {}; secs = []
    with Pool(nw) as pool:
        for i, (sid, st, sec) in enumerate(pool.imap_unordered(process, todo)):
            k = st.split(":")[0]; stat[k] = stat.get(k, 0) + 1
            if k == "done":
                secs.append(sec)
            elif k == "err":
                print(" ", sid, st, flush=True)
            if (i + 1) % 16 == 0 or i == 0:
                print(f"[{i+1}/{len(todo)}] {stat} mean {np.mean(secs) if secs else 0:.0f} s/subject", flush=True)
    print("status:", stat, flush=True)
    collect()


if __name__ == "__main__":
    main()
