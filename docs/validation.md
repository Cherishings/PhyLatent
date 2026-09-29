# Verification / 验证

Run these commands from the repository root after installation:

```bash
python -m unittest discover -s tests -v
python scripts/verify_checkpoints.py --load
```

The tests check configuration composition, losses, diagnostic edge cases and
paired statistics. The checkpoint command verifies file hashes and loads each
model on CPU, rejecting missing or unexpected parameters.

These checks have passed for the bundled code and checkpoints. They do not
establish a fresh full-training run or reproduction of the paper's evaluation
results. Full training and simulator-based evaluation require the runtime
dependencies, official data and a compatible GPU environment.

上述命令检查配置、损失、诊断统计和权重加载，已在随附代码与权重上通过。
完整训练与模拟器评估还需要运行依赖、官方数据和兼容的 GPU 环境；这些检查不等同于完整论文复现。
