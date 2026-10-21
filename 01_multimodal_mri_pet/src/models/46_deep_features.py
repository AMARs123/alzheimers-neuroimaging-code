"""
46_deep_features.py
-------------------
Pretrained 3D CNN features from T1 MRI (no training, no labels used).

Network: SFCN (Peng et al., 2021, Medical Image Analysis), brain-age model pretrained on UK Biobank,
weights from github.com/ha-ha-ha-han/UKBiobank_deep_pretrain (MIT licence).

Per subject:
  1. brain-extracted T1 (FastSurfer orig.mgz x mask.mgz)
  2. 12-parameter affine registration to the MNI152 (NLin6Asym, 1 mm) brain template with ANTs
  3. FSL/UK Biobank array convention (left-right flipped storage), centre crop 160x192x160, divide by mean
  4. forward pass: 64-d embedding (global average pooled last feature map) + predicted brain age

Outputs: ~/alz/p1/work/mni160/<NACCID>.npy (float16) and ~/alz/p1/work/deep_features_sfcn.csv
"""
import os, sys, tempfile, shutil
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, nibabel as nib

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
ALZ = Path.home() / "alz"; P1 = ALZ / "p1"; FS = ALZ / "fs_output"; PT = P1 / "pretrained"
OUT = P1 / "work" / "mni160"; OUT.mkdir(parents=True, exist_ok=True)
TEMPLATE = str(PT / "MNI152NLin6Asym_1mm_brain.nii.gz")


def prep(sid):
    out = OUT / f"{sid}.npy"
    if out.exists():
        return sid, "skip"
    tmp = tempfile.mkdtemp(prefix=f"mni_{sid}_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        import ants
        t1 = nib.load(str(FS / sid / "mri" / "orig.mgz")); m = nib.load(str(FS / sid / "mri" / "mask.mgz"))
        d = np.asarray(t1.dataobj).astype(np.float32) * (np.asarray(m.dataobj) > 0)
        f = os.path.join(tmp, "t1.nii.gz"); nib.save(nib.Nifti1Image(d, t1.affine), f)
        fixed = ants.image_read(TEMPLATE); moving = ants.image_read(f)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine", random_seed=1, verbose=False)
        w = os.path.join(tmp, "w.nii.gz"); ants.image_write(reg["warpedmovout"], w)
        a = np.asarray(nib.load(w).dataobj).astype(np.float32)        # RAS storage, 182x218x182
        if a.shape != (182, 218, 182):
            return sid, f"shape:{a.shape}"
        a = a[::-1]                                                    # FSL / UK Biobank convention: x stored right-to-left
        a = a[11:171, 13:205, 11:171]
        a = a / (a.mean() + 1e-6)
        tb = np.asarray(nib.load(TEMPLATE).dataobj)[::-1][11:171, 13:205, 11:171] > 0
        cc = float(np.corrcoef(a[::2, ::2, ::2].ravel(), tb[::2, ::2, ::2].ravel().astype(np.float32))[0, 1])
        np.save(out, a.astype(np.float16))
        return sid, f"done:{cc:.3f}"
    except Exception as e:
        return sid, "err:" + type(e).__name__ + ":" + str(e)[:120]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    lab = pd.read_csv(ALZ / "data" / "subset450_labels.csv")
    subs = sorted(s for s in lab.NACCID if (FS / s / "stats" / "aseg+DKT.VINN.stats").exists())
    print(f"{len(subs)} subjects", flush=True)
    nw = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) // 2)
    stat = {}; cc = {}
    with Pool(nw) as pool:
        for i, (sid, st) in enumerate(pool.imap_unordered(prep, subs)):
            k = st.split(":")[0]; stat[k] = stat.get(k, 0) + 1
            if k == "done":
                cc[sid] = float(st.split(":")[1])
            elif k != "skip":
                print(" ", sid, st, flush=True)
            if (i + 1) % 40 == 0:
                print(f"[{i+1}/{len(subs)}] {stat}", flush=True)
    print("registration:", stat, flush=True)

    import torch
    sys.path.insert(0, str(PT))
    from sfcn import SFCN
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    net = SFCN()
    sd = torch.load(PT / "sfcn_brain_age.p", map_location="cpu")
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}); net.eval()
    centers = torch.arange(42, 82).float() + 0.5
    rows = []
    with torch.no_grad():
        for i, sid in enumerate(subs):
            f = OUT / f"{sid}.npy"
            if not f.exists():
                continue
            x = torch.from_numpy(np.load(f).astype(np.float32))[None, None]
            fm = net.feature_extractor(x)
            emb = fm.mean(dim=(2, 3, 4))[0].numpy()
            logp = torch.log_softmax(net.classifier(fm), dim=1).reshape(-1)
            age = float((logp.exp() * centers).sum())
            r = {"NACCID": sid, "has_dl": 1, "dl_brainage": age}
            r.update({f"dl_emb_{j}": float(v) for j, v in enumerate(emb)})
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"inference {i+1}/{len(subs)}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(P1 / "work" / "deep_features_sfcn.csv", index=False)
    m = df.merge(lab, on="NACCID")
    print(f"wrote {len(df)} rows; predicted brain age: mean {m.dl_brainage.mean():.1f} sd {m.dl_brainage.std():.1f}; "
          f"correlation with NACCAGE (age at label visit) r={np.corrcoef(m.dl_brainage, m.NACCAGE)[0,1]:.3f}; "
          f"mean abs diff {np.abs(m.dl_brainage - m.NACCAGE).mean():.1f} y", flush=True)
    if cc:
        v = np.array(list(cc.values())); print(f"template overlap correlation: mean {v.mean():.3f} min {v.min():.3f}", flush=True)


if __name__ == "__main__":
    main()
