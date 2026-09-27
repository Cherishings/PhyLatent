# Data preparation / 数据准备

Obtain the task datasets from their official publishers and follow their
licenses. The original workspace used these LeWorldModel dataset repositories:

- Cube: https://huggingface.co/datasets/quentinll/lewm-cube
- TwoRooms: https://huggingface.co/datasets/quentinll/lewm-tworooms
- Reacher: https://huggingface.co/datasets/quentinll/lewm-reacher
- PushT: https://huggingface.co/datasets/quentinll/lewm-pusht

This package does not bundle raw data or claim a verified upstream dataset
revision. Record the downloaded revision and file SHA256 for your experiments.
原始数据不随代码发布；请遵守各数据集许可证，记录下载版本和文件哈希。

Place the extracted HDF5 files together in the directory passed as `data_root`
(training) or `--data-root` (evaluation/diagnostics):

```text
data/
  cube_single_expert.h5
  tworoom.h5
  reacher.h5
  pusht_expert_train.h5
```

For a locally downloaded archive / 解压已下载的本地压缩文件：

```bash
python scripts/prepare_data.py cube --source /path/to/cube_single_expert.h5.zst --data-root data
```

Training consumes `pixels`, `action`, and the physical-supervision field:
Cube/Reacher use `observation`, TwoRooms uses `proprio`, and PushT uses `state`.
Planning and diagnostics also need original episode/step indices, simulator
state and task-specific physical fields. Keep the official complete HDF5
files, not image/action-only conversions.
训练需要图像、动作与物理状态监督；规划/诊断还需原始 episode/step 和模拟器状态字段，勿裁成仅图像和动作。
