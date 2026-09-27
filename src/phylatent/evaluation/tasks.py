"""Task setup and shared planning preprocessing for PhyLatent."""
import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
TASKS = {'tworoom': {'dataset': 'tworoom.h5',
             'keys': ['action', 'proprio'],
             'world': {'env_name': 'swm/TwoRoom-v1'},
             'callables': [{'method': '_set_state',
                            'args': {'state': {'value': 'proprio'}}},
                           {'method': '_set_goal_state',
                            'args': {'goal_state': {'value': 'goal_proprio'}}}]},
 'reacher': {'dataset': 'reacher.h5',
             'keys': ['action'],
             'world': {'env_name': 'swm/ReacherDMControl-v0', 'task': 'qpos_match'},
             'callables': [{'method': 'set_state',
                            'args': {'qpos': {'value': 'qpos'},
                                     'qvel': {'value': 'qvel'}}},
                           {'method': 'set_target_qpos',
                            'args': {'target_qpos': {'value': 'goal_qpos'}}}]},
 'pusht': {'dataset': 'pusht_expert_train.h5',
           'keys': ['action', 'proprio', 'state'],
           'world': {'env_name': 'swm/PushT-v1'},
           'callables': [{'method': '_set_state',
                          'args': {'state': {'value': 'state'}}},
                         {'method': '_set_goal_state',
                          'args': {'goal_state': {'value': 'goal_state'}}}]},
 'cube': {'dataset': 'cube_single_expert.h5',
          'keys': ['action'],
          'world': {'env_name': 'swm/OGBCube-v0',
                    'env_type': 'single',
                    'ob_type': 'states',
                    'multiview': False,
                    'width': 224,
                    'height': 224,
                    'visualize_info': False,
                    'terminate_at_goal': True},
          'callables': [{'method': 'set_state',
                         'args': {'qpos': {'value': 'qpos'},
                                  'qvel': {'value': 'qvel'}}},
                        {'method': 'set_target_pos',
                         'args': {'cube_id': {'value': 0, 'in_dataset': False},
                                  'target_pos': {'value': 'goal_privileged_block_0_pos'},
                                  'target_quat': {'value': 'goal_privileged_block_0_quat'}}}]}}

def image_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**IMAGENET),
        transforms.Resize(size=224),
    ])

def fit_processors(dataset, keys):
    process = {}
    for key in keys:
        if key == "pixels":
            continue
        values = dataset.get_col_data(key)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(values)
        process[key] = scaler
        if key != "action":
            process[f"goal_{key}"] = scaler
    return process

def sample_tasks(dataset, count, offset, seed):
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_ids = dataset.get_col_data(ep_key)
    steps = dataset.get_col_data("step_idx")
    unique = np.unique(ep_ids)
    lengths = {int(ep): int(steps[ep_ids == ep].max()) + 1 for ep in unique}
    valid = np.flatnonzero(steps <= np.array([lengths[int(ep)] - offset - 1 for ep in ep_ids]))
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} valid starts for {count} evaluations")
    picked = np.sort(np.random.default_rng(seed).choice(valid, size=count, replace=False))
    rows = dataset.get_row_data(picked)
    return rows[ep_key].tolist(), rows["step_idx"].tolist()
