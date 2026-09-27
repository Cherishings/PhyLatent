"""CPU-only checks for base losses, ablations and diagnostic semantics."""
import unittest
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from phylatent.losses import normalized_mse, action_separation_loss
from phylatent.training import apply_ablation
from phylatent.diagnostics.metrics import invariance_failures, identifiability_failures, hierarchical_paired_bootstrap
from phylatent.diagnostics.ordering import model_ordering_arrays, hierarchical_pooled_paired_bootstrap
from phylatent.diagnostics.reporting import summarize
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_task_and_ablation_composition(self):
        variants=["full","wo_psg","wo_fra","wo_svip","wo_casp","wo_ld",
            "wo_physical_group","wo_invariance_group","wo_counterfactual_group"]
        keys={"cube":"observation","tworoom":"proprio","reacher":"observation","pusht":"state"}
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/"configs/train")):
            for task,key in keys.items():
                for variant in variants:
                    cfg=compose(config_name=task,overrides=[f"ablation={variant}"])
                    apply_ablation(cfg)
                    self.assertEqual(cfg.loss.state.key,key)
                    self.assertEqual(cfg.data.dataset.num_steps,4)
                    self.assertEqual(cfg.task,task)
                    for loss in cfg.ablation.disable:self.assertEqual(cfg.loss[loss].weight,0)
                    self.assertEqual(cfg.optimizer.lr,5e-5)

    def test_normalized_mse_is_feature_mean(self):
        a=torch.tensor([[1.,0.]])
        b=torch.tensor([[0.,1.]])
        self.assertEqual(normalized_mse(a,b).item(),1.)
        self.assertEqual(normalized_mse(a*10,b*3).item(),1.)

    def test_action_separation_detaches_clean_prediction(self):
        a=torch.zeros(2,3,4,requires_grad=True)
        b=torch.full((2,3,4),0.01,requires_grad=True)
        action=torch.zeros(2,3,2)
        counter=torch.ones_like(action)
        cfg=OmegaConf.create({"loss":{"action_separation":{"margin_scale":1.,"max_gap":2.}}})
        loss=action_separation_loss(a,b,action,counter,cfg)
        loss.backward()
        self.assertIsNone(a.grad)
        self.assertGreater(b.grad.abs().sum().item(),0)

    def test_ordering_ties_and_invalid_values(self):
        eligible,failure=invariance_failures(np.array([1.,2.]),np.array([2.,1.]),np.array([2.,1.]),np.array([2.,2.]))
        self.assertEqual(eligible.tolist(),[True,False])
        self.assertTrue(failure[0])
        self.assertEqual(identifiability_failures(np.array([1.,1.]),np.array([1.,np.nan])).tolist(),[True,True])

    def test_cf_ties_fail_and_physical_ties_are_ineligible(self):
        p=np.array([[[[0.,1.,2.,3.]]]])
        ids=np.argsort(p,axis=-1)
        arrays=model_ordering_arrays(p,np.ones_like(p),p,ids[...,:1],ids[...,-1:])
        self.assertTrue(arrays["eligible"].all())
        self.assertTrue(arrays["failure"].all())
        tied=model_ordering_arrays(np.ones_like(p),np.ones_like(p),np.ones_like(p),ids[...,:1],ids[...,-1:])
        self.assertFalse(tied["eligible"].any())

    def test_paired_bootstrap_constant_difference(self):
        baseline=np.zeros((2,3));method=np.ones((2,3))
        stats=hierarchical_paired_bootstrap(baseline,method,resamples=20,seed=10)
        self.assertEqual(stats,{"difference":1.,"ci95_low":1.,"ci95_high":1.})
        eligible=np.ones((2,3,2,2),bool)
        stats=hierarchical_pooled_paired_bootstrap(eligible,np.zeros_like(eligible),eligible,resamples=20,seed=10)
        self.assertEqual(stats,{"difference":1.,"ci95_low":1.,"ci95_high":1.})

    def test_report_uses_common_cf_eligibility(self):
        physical=np.broadcast_to(np.arange(21,dtype=float),(2,3,3,21)).copy()
        observed=physical.copy();observed[..., -1]=0
        branches={"reference":{"counterfactual_physical_distance":physical,"counterfactual_true_distance":observed,"counterfactual_pred_distance":physical},
            "phylatent":{"counterfactual_physical_distance":physical,"counterfactual_true_distance":physical,"counterfactual_pred_distance":np.ones_like(physical)}}
        f={"invariance_primary":np.zeros((2,3)),"invariance_clean_coverage":np.ones((2,3)),"identifiability_primary":np.zeros((2,3))}
        report=summarize({"reference":f,"phylatent":f},branches,20)
        self.assertEqual(report["models"]["phylatent"]["cf_ordering"]["h5"]["collapse_percent"],100.)
        self.assertEqual(report["paired"]["cf_h5"]["common_eligible_contrasts"],2*3*4*5)
        self.assertEqual(report["paired"]["cf_h5"]["difference"],100.)

if __name__=="__main__":unittest.main()
