# Videos / 视频说明

[English README](../README.md) · [中文首页](../README.zh-CN.md)

## Complete task demonstrations / 完整任务演示

All four task videos use the release checkpoints, verified by SHA256 before inference.
The planner retains CEM 300 candidates/top-30, 10 iterations (30 for PushT), horizon 5,
action block 5 and receding horizon 5. Each episode uses a fixed later goal from the same
source episode. Demonstration budgets are at most 400 environment steps, ending at success; Cube uses a 100-step budget and continues autonomously after first goal success until three consecutive released, near-table, low-speed observations.
These are selected demonstrations, not a new benchmark success-rate claim.

四任务视频均使用发布权重，运行前核验 SHA256。采用原规划设置，另选较早起点与同 episode
较晚目标，执行上限不超过 400 环境步，成功后结束；本版 Cube 录制上限为 100 步，首次达标后继续自主执行，直到连续三帧近桌面、低速且夹爪接触读数低于 0.1。它们是选取的演示，不替代论文原协议的成功率。

- Cube: 48 steps, initial dataset step 26, 20 source fps, 0.25x presentation. 中文：[完整视频](../assets/videos/cube_full_zh.mp4) / [English](../assets/videos/cube_full_en.mp4).
- TwoRooms: 45 steps, initial dataset step 0, 10 source fps, 0.5x presentation. 中文：[完整视频](../assets/videos/tworoom_full_zh.mp4) / [English](../assets/videos/tworoom_full_en.mp4).
- Reacher: 25 steps, initial dataset step 0, nominal 25 source fps, 0.1x presentation. 中文：[完整视频](../assets/videos/reacher_full_zh.mp4) / [English](../assets/videos/reacher_full_en.mp4).
- PushT: 50 steps, initial dataset step 0, 10 source fps, 0.5x presentation. 中文：[完整视频](../assets/videos/pusht_full_zh.mp4) / [English](../assets/videos/pusht_full_en.mp4).

Each full presentation includes three seconds of initial/final hold. Slowing uses repeated
source frames only, with no generated intermediate motion. Source images are 224×224;
the 720p layout enlarges them without claiming additional image detail. Reacher uses
qpos matching; its nominal step is two 0.02-second physics steps and success may stop the
last repeat early. The target image is the authoritative displayed goal; decorative target
markers in source renders should not substitute for it.

每条完整展示片在开头和结尾各停留三秒。慢放只重复原始帧，不生成中间动作。
原始图像为 224×224，720p 版为排版放大，并非更高分辨率的新观测。
Reacher 的任务为目标关节姿态匹配。所有任务以旁边的固定目标图为目标参照。

## Cube chapters / Cube 三章专题

[English full film](../assets/videos/cube_three_cases_en.mp4) · [中文完整专题](../assets/videos/cube_three_cases_zh.mp4)

1. **Invariance / 不变性**: case 20164, checker appearance perturbation. LeWM fails
   within 50 steps; the matched PhyLatent run succeeds at step 19. The video shows both
   the unmodified simulator scene and the perturbed agent input.
2. **Distinguishability / 可区分性**: case 82020 supplies the full-state diagnostic
   and complete LeWM failure. A separate planning comparison uses task-goal
   recovery case 1201110, where LeWM fails and PhyLatent succeeds at step 20. Its object-to-goal
   scoring improves, but its full-state distance ordering fails the original diagnostic
   eligibility: it is not evidence of same-case recovery of that diagnostic. 原诊断与恢复对照
   来自不同案例，不能拼成一条轨迹或声称已经证明同一项诊断被解决。
3. **Counterfactual / 反事实**: case 891917. Encoded actual futures prefer a good plan,
   while LeWM's predicted scores reverse that preference. The matched PhyLatent run
   succeeds at step 24; LeWM fails within 50. Future images are actual simulator
   observations, not decoded model predictions.

Paired runs use identical recorded physical initial states, goals and initial CEM candidate
pools. Archived first-pool scores for the two newly recovered pairs were reproduced with
both frozen checkpoints (maximum absolute error 0). All paired policies choose their own
actions; the successful paths are not human action-replacement interventions.
Diagnostic branch endpoints are explanatory counterfactual executions, labelled separately
from the autonomous rollouts. These selected comparisons do not isolate any single loss
as the sole cause of improvement.

每组自主规划对照均核对了物理初态、目标与首次 CEM 候选池。新增两组回放的首轮评分已用
冻结权重复算，两模型最大绝对误差均为 0。专题中的自主成功轨迹没有人工替换动作。
诊断中的动作分支终点是另行执行的解释性分支，不冒充主回放或模型生成图像。

Video hashes and execution metadata are listed in the [media manifest](../assets/videos/manifest.json).
All MP4 files are silent H.264/yuv420p; captions are burned into edited versions.
完整片无音轨，便于自行配音；公开版字幕已嵌入画面。

Current task clips were reselected for clearer motion and fewer unnecessary reversals. All four task clips were replaced; the Cube case-study film is unchanged.
本版重新筛选四任务成功轨迹，优先展示清晰、少回摆的过程；四个任务均已替换，Cube 专题保持不变。

The task-video footer was removed in both languages. The Cube demonstration now begins before gripper contact and retains all 48 autonomous steps through release near the target. Original goal success first occurs at step 16; the recording ends at step 48 after three low-contact, near-table, low-speed frames. No action is replaced. The recording stop marker is distinct from the original environment success flag; see [placement audit](../assets/videos/cube_placement_audit.json).
中英文四任务单片均已删除底部说明。Cube 改用夹取前起步的轨迹，保留全部 48 步。第 16 步首次满足原环境目标阈值，第 48 步满足连续三帧放下的录制停止条件；没有人工替换动作。记录停止标记与原环境成功标记分别保存，见放置核验文件。

## Display camera for case 1201110 / 案例 1201110 的展示视角

The paired execution panels use a shared wider camera, shifted so the arm and gripper move left and down in the frame. Every frame is re-rendered from archived simulator states, with zero environment steps and unchanged actions, policy observations and metrics. Goal thumbnails retain the original observations. The display camera is documented here; [camera provenance](../assets/videos/cube_display_camera.json) records parameters and source hashes.

两模型的配对执行画面使用同一放宽视角，主体向左下移动，为右侧夹爪和上方机械臂留出空间。所有画面均从原存档物理状态重新渲染，没有执行新动作；模型输入、目标小图与结果保持原样。

## Initial and goal views / 初始画面与目标画面

The introduction shows the case’s initial observation and desired goal observation. These are not separate model outcomes. The three introductions retain their titles and identify the collapse category, with a centered image-and-text layout.

开场左图是案例开始时的观测，右图是希望到达的目标观测；它们不是 LeWM 和 PhyLatent 各自的执行结果。三章开场保留主标题，用副标题标明坍缩类别，删除三行通用说明，图文上移居中。

## Guided stories and transitions / 分步讲解与转场

The three original opening cards are retained. Explanations now introduce the real outcomes before showing their scores: A is the worse outcome, B is the better outcome, and lower model scores receive priority. Scores should be compared between A and B within a panel, not across panels. Counterfactual B is an additional proposal used to probe scoring and was not in LeWM’s original candidate pool; the video does not claim LeWM chose A over an available B. The distinguishability diagnosis and task-success pair are separate cases, documented above; their case IDs and the transition explanation are omitted from the video.

三个开场画面保留。后续先看真实执行结果与厘米距离，再看模型评分，并始终用 A、B 对应同一方案。每页只比较 A 与 B 的分数，不跨页比较大小。反事实 B 是额外加入的比较方案，不在 LeWM 原候选池中；结论是预测评分对两方案的偏好反转，不能说 LeWM 当时在原候选池中舍弃了 B。可区分性的诊断与成功对照来自上文所列的不同案例；视频省略案例编号与换案例说明。

Edited chapters are 30 fps with 0.4-second smooth cross-dissolves between held boundary frames. These transitions add editorial time only. Every recorded motion step remains visible at the stated playback speed; there is no synthetic motion interpolation. Raw recordings and the four task demonstrations are unchanged.

成片以 30 fps 输出，段落之间添加 0.4 秒柔和淡化。转场只增加剪辑时间，不删减控制步、不生成新的机器人动作；四任务视频和原始素材保持原样。

## Caption layout / 画面文字

Case IDs, tutorial prompts and standalone closing cards are omitted. Each chapter ends with a two-second hold on the paired execution result. Original opening cards, scores and full recorded trajectories are retained.

视频已删除案例编号、引导口号和独立总结卡片，每章直接停在配对执行结果上两秒。开场、评分与完整记录轨迹保留。
