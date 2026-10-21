"""
20_run_fastsurfer_batch.py
--------------------------
Har subject ke liye:
 1. MPRAGE/T1 series dhoondo (FLAIR skip)
 2. dcm2niix -> proper .nii (sahi affine)
 3. FastSurfer segmentation (GPU) -> region stats
Resume-friendly: jo done (stats file hai) skip karta.

Output: ~/alz/fs_output/<NACCID>/stats/aseg+DKT.VINN.stats  har subject ka
"""
import os, glob, subprocess, sys
from pathlib import Path

HOME=Path.home(); ALZ=HOME/"alz"
MRI_ROOT=ALZ/"data"/"MRI"/"SCAN"
NII_DIR=ALZ/"nii_all"; NII_DIR.mkdir(exist_ok=True)
FS_OUT=ALZ/"fs_output"
FS=ALZ/"FastSurfer"/"run_fastsurfer.sh"
LABELS=ALZ/"data"/"subset450_labels.csv"

import pandas as pd
lab=pd.read_csv(LABELS)
subjects=sorted(set(os.listdir(MRI_ROOT)) & set(lab["NACCID"]))
# parallel: START/END env se subjects baanto (multiple jobs alag range karein)
START=int(os.environ.get("START",0))
END=int(os.environ.get("END",len(subjects)))
subjects=subjects[START:END]
print(f"Processing subjects [{START}:{END}] = {len(subjects)}",flush=True)

def find_mprage(sid):
    """MPRAGE/T1 series folder dhoondo (FLAIR/other skip)."""
    subj=MRI_ROOT/sid
    best=None; best_score=-1
    for series in os.listdir(subj):
        low=series.lower()
        score=0
        if any(k in low for k in ["mprage","t1","spgr","fspgr"]): score+=100
        if "flair" in low or "t2" in low or "dwi" in low or "dti" in low: score-=100
        dcms=glob.glob(str(subj/series/"**"/"*.dcm"),recursive=True)
        score+=len(dcms)*0.001
        if dcms and score>best_score:
            best_score=score; best=os.path.dirname(dcms[0])
    return best

done=0; skip=0; fail=0
for i,sid in enumerate(subjects):
    stats=FS_OUT/sid/"stats"/"aseg+DKT.VINN.stats"
    if stats.exists():
        skip+=1; continue
    try:
        t1dir=find_mprage(sid)
        if not t1dir:
            print(f"[{i+1}] {sid} no MPRAGE",flush=True); fail+=1; continue
        nii=NII_DIR/f"{sid}.nii.gz"
        if not nii.exists():
            subprocess.run(["dcm2niix","-z","y","-o",str(NII_DIR),"-f",sid,t1dir],
                           capture_output=True,timeout=120)
        if not nii.exists():
            # dcm2niix ne alag naam banaya ho
            cand=list(NII_DIR.glob(f"{sid}*.nii.gz"))
            if cand: nii=cand[0]
            else: print(f"[{i+1}] {sid} nii fail",flush=True); fail+=1; continue
        # FastSurfer seg
        r=subprocess.run(["bash",str(FS),"--t1",str(nii),"--sid",sid,"--sd",str(FS_OUT),
                          "--seg_only","--device","cuda","--no_cereb","--threads","4"],
                         capture_output=True,text=True,timeout=600)
        if stats.exists():
            done+=1; print(f"[{i+1}/{len(subjects)}] {sid} DONE",flush=True)
        else:
            fail+=1; print(f"[{i+1}] {sid} seg fail: {r.stderr[-200:]}",flush=True)
    except subprocess.TimeoutExpired:
        print(f"[{i+1}] {sid} timeout",flush=True); fail+=1
    except Exception as e:
        print(f"[{i+1}] {sid} ERR {str(e)[:100]}",flush=True); fail+=1

print(f"\nDone {done}, skip {skip}, fail {fail}",flush=True)
