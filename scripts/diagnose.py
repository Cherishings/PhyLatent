#!/usr/bin/env python3
"""Collect and summarize PhyLatent collapse diagnostics."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["cube","tworoom","reacher","pusht"],required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--reference-dir",type=Path,help="Optional reference weights for paired comparisons")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seeds",nargs="+",type=int,default=[4100,4101,4102])
    p.add_argument("--frame-anchors",type=int,default=1000)
    p.add_argument("--frame-pool",type=int,default=4096)
    p.add_argument("--references-per-bin",type=int,default=64)
    p.add_argument("--action-anchors",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--bootstrap-resamples",type=int,default=10000)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    import h5py
    import hdf5plugin
    import numpy as np
    import torch
    from phylatent.models.loading import load_model
    from phylatent.diagnostics.collection import TASKS, make_physical_calibration, frame_stage, counterfactual_stage
    from phylatent.diagnostics.metrics import sha256
    from phylatent.diagnostics.reporting import summarize, json_safe
    if args.smoke:
        args.seeds=[4100];args.frame_anchors=8;args.frame_pool=256
        args.references_per_bin=4;args.action_anchors=2;args.bootstrap_resamples=100
    dataset=args.data_root/TASKS[args.task]["dataset"]
    models={"phylatent":load_model(args.model_dir,args.device)}
    if args.reference_dir:
        models={"reference":load_model(args.reference_dir,args.device),**models}
    args.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(dataset,"r") as f:
        calibration=make_physical_calibration(f,args.task,200000,77123)
        frame,frame_plan=frame_stage(f,args.task,models,args.seeds,args.frame_anchors,args.frame_pool,
            args.references_per_bin,calibration,args.batch_size,torch.device(args.device))
        branches,branch_plan=counterfactual_stage(f,args.task,TASKS[args.task],models,args.seeds,
            args.action_anchors,calibration,args.batch_size,torch.device(args.device))
    for label in models:
        np.savez_compressed(args.output/f"raw_{label}.npz",**frame[label],**branches[label])
    report=summarize(frame,branches,args.bootstrap_resamples)
    report["manifest"]={"task":args.task,"dataset":str(dataset),"seeds":args.seeds,
        "parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "weights_sha256":{"phylatent":sha256(args.model_dir/"weights.pt"),
            **({"reference":sha256(args.reference_dir/"weights.pt")} if args.reference_dir else {})}}
    report["frame_plan"]=frame_plan;report["counterfactual_plan"]=branch_plan
    (args.output/"summary.json").write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(json_safe(report["models"]),indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
