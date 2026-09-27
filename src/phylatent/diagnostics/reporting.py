"""Summarize final invariance, distinguishability, and CF ordering metrics."""
import numpy as np
from .metrics import hierarchical_paired_bootstrap
from .ordering import model_ordering_arrays, hierarchical_pooled_paired_bootstrap


def summarize(frame, branches, bootstrap_resamples=10000):
    result = {"models": {}, "paired": {}}
    arrays = {}
    for label, values in branches.items():
        physical = values["counterfactual_physical_distance"]
        order = np.argsort(physical, axis=-1, kind="stable")
        arrays[label] = model_ordering_arrays(values["counterfactual_true_distance"],
            values["counterfactual_pred_distance"], physical, order[..., :5], order[..., -5:])
        f = frame[label]
        result["models"][label] = {
            "inv_percent": 100 * float(np.nanmean(f["invariance_primary"])),
            "inv_coverage_percent": 100 * float(np.nanmean(f["invariance_clean_coverage"])),
            "dist_percent": 100 * float(np.nanmean(f["identifiability_primary"])),
            "cf_ordering": {}}
        for i, h in enumerate((1, 3, 5)):
            a = arrays[label]; eligible = a["eligible"][:, :, i]; failure = a["failure"][:, :, i]
            contrast = a["physical_contrast"][:, :, i]
            result["models"][label]["cf_ordering"][f"h{h}"] = {
                "collapse_percent": 100*float((eligible & failure).sum()/eligible.sum()) if eligible.any() else None,
                "eligible_contrasts": int(eligible.sum()),
                "coverage_percent": 100*float(eligible.sum()/contrast.sum()) if contrast.any() else None}
    if "reference" in frame:
        for i, (name,key) in enumerate((("inv", "invariance_primary"),("dist", "identifiability_primary"))):
            stats = hierarchical_paired_bootstrap(frame["reference"][key], frame["phylatent"][key],
                resamples=bootstrap_resamples, seed=92000+i)
            result["paired"][name] = {k:100*v for k,v in stats.items()}
        a,b = arrays["reference"], arrays["phylatent"]
        for i,h in enumerate((1,3,5)):
            common = a["eligible"][:,:,i] & b["eligible"][:,:,i]
            n=int(common.sum())
            stats=hierarchical_pooled_paired_bootstrap(common,a["failure"][:,:,i],b["failure"][:,:,i],
                resamples=bootstrap_resamples,seed=93000+h)
            result["paired"][f"cf_h{h}"]={"common_eligible_contrasts":n,
                "reference_percent":100*float((common&a["failure"][:,:,i]).sum()/n) if n else None,
                "phylatent_percent":100*float((common&b["failure"][:,:,i]).sum()/n) if n else None,
                **{k:100*v for k,v in stats.items()}}
    return result


def json_safe(value):
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    return value
