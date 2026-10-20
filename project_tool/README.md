# project_tool

本机（Yahboom Jetson Orin NX，`jetson@yahboom`）上调试底盘、脖子和相机用的小工具。

所有脚本都在 `lerobot_env` 里运行：

```bash
conda activate lerobot_env
cd ~/workspaces/FXLerobot/FXlerobot/project_tool
```

带画面的脚本会弹窗口，要在机器人桌面的终端里运行。SSH 进来时脚本默认用 `DISPLAY=:0`，窗口出现在机器人屏幕上。

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `lerobot_kiwi_base.py` | 键盘遥控三轮全向底盘 |
| `lerobot_kiwi_base_neck.py` | 键盘遥控底盘和脖子 |
| `head_camera_test.py` | 头部左右眼实时画面，带辅助十字和区域框 |
| `wrist_camera_test.py` | 左右腕部相机实时画面，显示帧率，可对调左右 |
| `calibrate_head_cameras.py` | 头部双目 ChArUco 标定：内参、双目外参、相机到底盘外参 |
| `validate_head_extrinsic.py` | 校验相机到底盘的外参：双目一致性、竖放板子检查、地面网格叠加和点击测距 |
| `nlv_nav.py` | 短距离语言导航：通义千问看头部画面找目标，底盘走到目标跟前 |
| `save_pose.py` | 记录两只从臂的当前关节位置，之后读回来恢复到这个姿势 |

## 设备名

串口和相机都用 udev 固定了名字，规则在 `../working_sh/`，用 `../working_sh/install_udev_rules.sh` 安装。

左右指的是机器人自身的左右：站在机器人背后、和它朝同一方向看。

| 名字 | 设备 | USB 口 |
|---|---|---|
| `/dev/base_neck` | 底盘舵机 ID 7/8/9 和脖子舵机 ID 10/11 | 按序列号绑定 |
| `/dev/cam_front_left` | 头部左眼 C920 | 2.4 |
| `/dev/cam_front_right` | 头部右眼 C920 | 2.2 |
| `/dev/cam_wrist_left` | 左腕相机 | 2.3.4 |
| `/dev/cam_wrist_right` | 右腕相机 | 2.1.3 |

两台 C920 没有 USB 序列号，两台腕部相机的序列号相同，所以相机只能按 USB 口绑定。**相机线不能换口**，换了口名字会跟着口走，指到另一台相机上。

`/dev/videoN` 的编号每次开机都可能变，脚本里不要写 `VideoCapture(0)` 这种编号。

## 底盘和脖子遥控

```bash
python lerobot_kiwi_base.py         # 只控底盘
python lerobot_kiwi_base_neck.py    # 底盘 + 脖子
```

| 按键 | 动作 |
|---|---|
| `w` / `s` | 前进 / 后退 |
| `a` / `d` | 左移 / 右移 |
| `z` / `x` | 左转 / 右转 |
| `r` / `f` | 速度档位升 / 降（仅 `lerobot_kiwi_base.py`） |
| `r` / `t` | 脖子 yaw +5° / -5°（仅 `lerobot_kiwi_base_neck.py`） |
| `v` / `b` | 脖子 pitch +5° / -5°（仅 `lerobot_kiwi_base_neck.py`） |
| `q` | 停车并退出 |

- 按键在运行脚本的终端里读取。
- 底盘速度档位：慢 0.1 m/s、30°/s；中 0.25 m/s、60°/s（默认）；快 0.4 m/s、90°/s。
- `lerobot_kiwi_base_neck.py` 一启动就把脖子发到 yaw=0、pitch=0。
- `lerobot_kiwi_base_neck.py` 里按 `r` 除了转脖子，也会把底盘速度升一档，而且没有降档键。
- 提示 `串口已被占用` 时，先结束旧进程：`pkill -f lerobot_kiwi_base`。

## 相机查看

```bash
python head_camera_test.py     # 头部左右眼，ESC 退出
python wrist_camera_test.py    # 腕部左右，s 对调左右，q / ESC 退出
```

## 头部双目标定

标定板：ChArUco，5×7 个方格，`DICT_5X5_50`，方格边长 28.2 mm，码边长约 20 mm。这些都是脚本默认值，换板子时用 `--square-mm`、`--marker-mm` 改。

数据和结果保存在 `~/workspaces/FXLerobot/FXlerobot/calib/head/`，可用 `--data` 换目录。

### 1. 采图

```bash
python calibrate_head_cameras.py capture
```

- 拿着板子在两只眼前换位置、换倾斜角，每个位置停稳约 1 秒，脚本会自动存一对图。窗口底部提示 `HOLD STILL` 时要拿稳，提示 `MOVE TO NEW POSE` 时要换位置。
- 左右角点数显示绿色（不少于 8 个）才算有效。
- 手动保存：先点一下图像窗口，再按空格或 `s`；或者在终端里按回车。退出：ESC、`q` 或 `Ctrl+C`。
- 存 25 到 30 对。大部分照片要让两只眼同时看到板；两只眼各向内倾约 9°，共同视野集中在正前方中间。
- 尽量让板子贴近画面四角和左右边缘，这些照片对畸变最有用。
- 加 `--no-auto`（`capture --no-auto`）可关闭自动保存，只按键保存。

### 2. 算内参和双目外参

```bash
python calibrate_head_cameras.py cali_intri
```

结果写入 `head_cameras.yaml`：

- `left` / `right`：相机矩阵 `K`、畸变 `dist`、重投影误差 `rms`。
- `stereo`：`R_right_left`、`T_right_left_m`，满足 `X_right = R · X_left + T`。另外记录了间距 `baseline_mm` 和光轴夹角 `optical_axis_angle_deg`。

怎么判断结果能用：

- 每只眼 `rms` 小于 0.5 到 0.6 px，双目 `rms` 小于 0.8 px。
- `baseline_mm` 接近用尺子量的两颗镜头中心距。
- 脚本会列出误差最大的三张照片；某张明显偏大时，删掉那一对图后重算。
- 默认把畸变系数 k3 固定为 0。只有在照片里很多角点贴近画面边缘时，才加 `--fit-k3`，否则 k3 会乱拟合。

当前结果（2026-10-13，32 对）：

| | 左眼 | 右眼 |
|---|---|---|
| rms | 0.50 px | 0.54 px |
| fx / fy | 608.4 / 606.1 | 609.4 / 606.1 |
| cx / cy | 328.0 / 249.5 | 353.0 / 249.7 |

双目：rms 0.62 px，间距 142.7 mm，光轴夹角 18.5°。

`head_cameras_OLD_NAMING*.yaml` 是左右改名前的旧结果，左右是反的，不要用。

### 3. 相机到底盘的外参

底盘坐标系和 `LeKiwiBase` 一致：x 朝前，y 朝左，z 朝上，原点在底盘中心正下方的地面上。

外参用竖放的板子标。相机离地约 65 cm、向下看约 33°，A4 板平放在地上时离得太远、看起来太扁，角点检测不出来。

1. 固定脖子，记下当前的 yaw 和 pitch。用 `lerobot_kiwi_base_neck.py` 回到 yaw=0、pitch=0 最容易复现。
2. 板子贴在纸箱、书或墙上，**竖直**立在车头正前方约 0.6–0.8 m 处，正面朝向机器人，底边可以直接落地。两只眼都要看到整块板，离画面边缘留点余量。
3. 先跑一次，看脚本说原点是哪个角：

   ```bash
   python calibrate_head_cameras.py cali_extri
   ```

   终端会打印 `Measure from the bottom robot-left corner ...` 这样的提示，意思是：站在机器人背后、和机器人同朝向看过去，棋盘格图案（不是纸边）的左下角是原点。板子怎么转都行，脚本会自动判断方向。图片保存为 `extrinsic_left_*.png` 和 `extrinsic_right_*.png`，红色是板子的 x 轴，绿色是 y 轴，两条轴的起点就是原点。

4. 量这个角点：在底盘中心前方多远（x）、偏左多远（y，偏右为负）、离地多高（z），以及板面朝向相对车头正前方逆时针转了多少度（正对机器人就是 0）。带上这些值再跑：

   ```bash
   python calibrate_head_cameras.py cali_extri \
       --board-x-mm 700 --board-y-mm 70 --board-z-mm 25 --board-yaw-deg 0 \
       --neck-yaw 0 --neck-pitch 0
   ```

   数字只是示例。板子必须竖直，前倾或后仰 1° 大约让相机高度差 1 cm；可以用手机水平仪靠在板子上确认。

结果写入 `head_extrinsic.yaml`：每只眼的 `T_base_cam`（4×4，相机坐标转到底盘坐标）、`pitch_down_deg`、`yaw_left_deg`，以及标定时的脖子角度和板子位置。

一张图算出的相机位置误差在 1 cm 左右。最好换两三个位置各标一次，看结果是否一致；也可以用卷尺量镜头离地高度对比。外参只在记录的那组脖子角度下成立，脖子一动就要按角度换算。

### 4. 校验外参

脖子要和标定外参时的角度相同，脚本启动时会打印标定时记录的角度。

```bash
python validate_head_extrinsic.py              # 实时窗口
python validate_head_extrinsic.py --headless   # 只打印一次结果
```

它做三项检查，每项打印 `OK` 或 `CHECK`：

| 检查 | 怎么做 | OK 的标准 |
|---|---|---|
| 双目一致性 | 用左右眼外参反推两眼相对位置，和 `head_cameras.yaml` 的双目结果比较，不用摆板子 | 转角差 < 1°，位置差 < 10 mm |
| 竖放检查 | 板子竖着放，**换一个和标定外参时不同的距离或角度**，两只眼分别算板子在底盘坐标下的位置 | 板子前后倾斜 < 1.5°，左右眼算出的板子位置相差 < 8 mm |
| 和卷尺比较（可选） | 加 `--board-x-mm`、`--board-y-mm`、`--board-z-mm`、`--board-yaw-deg`，量法和第 3 步相同 | 位置差 < 20 mm，朝向差 < 1.5° |

实时窗口里：

- 地面画了底盘坐标下的网格，每格 10 cm。黄色粗线是 x = 0.5 m、1.0 m、1.5 m，红线是车头正前方（y = 0）。可以在地上用胶带贴出 0.5 m、1.0 m 的位置，看是否和黄线重合。
- 点击画面里的地面，终端打印这一点在底盘坐标下的 x、y，两只眼里都会画上紫色十字。如果外参对，十字在另一只眼里也会落在同一处地面上。
- 空格、`s` 或终端里按回车，把这一次的竖放检查结果追加到 `head_extrinsic.yaml` 同目录下的 `extrinsic_check.csv`。ESC 或 `q` 退出。

用保存好的图片检查时，加 `--left-image a.png --right-image b.png`。


## 短距离语言导航 `nlv_nav.py`

用一句话指定目标，机器人走到头部相机看得见的地面目标跟前（约 3 m 以内）。左眼拍一张图发给通义千问 Qwen-VL，模型返回目标的边界框；之后由 ViT 跟踪器每帧跟住这个框，框的下边中点就是目标贴地处，用 `head_extrinsic.yaml` 换算成底盘坐标下的地面点，每帧据此更新转向和前进速度。全向轮在地毯上打滑、走不直，靠每帧重新计算来纠正。跟丢或每走 5 s，停下重新问一次模型。目标离机器人 0.6 m，或目标底部已到画面下沿（头部相机看到的地面从约 0.52 m 开始）时停止。

跟踪模型放在 `~/workspaces/FXLerobot/FXlerobot/models/object_tracking_vittrack_2023sep.onnx`（来自 opencv_zoo）。

先在 `~/.bashrc` 里设置 key，再开一个新终端：

```bash
export DASHSCOPE_API_KEY=sk-...
```

```bash
python nlv_nav.py "去红色箱子旁边" --dry-run   # 只定位一次，不动电机
python nlv_nav.py "去红色箱子旁边"             # 开始走，Ctrl-C 停车
```

- 每次问模型的画面存为 `nav_logs/<时间>/lookXX.jpg`，跟踪过程每秒存一张 `lookXX_NNNN.jpg`，红框是跟踪到的目标、红十字是贴地点。先用 `--dry-run` 看红框下边是否落在目标底部、终端打印的 x、y 是否和卷尺对得上。
- 启动时把脖子转到标定外参时的角度（`head_extrinsic.yaml` 里记录的 yaw、pitch）。底盘串口不能被遥控脚本占用。
- 默认模型 `qwen3-vl-plus`，它返回 0–1000 归一化坐标。换成 Qwen2.5-VL 系列（如 `qwen-vl-max`）时加 `--coord abs`。
- 停止距离 `--stop-m`（默认 0.6），速度上限 `--lin`、`--rot`（默认 0.15 m/s、30°/s），重新确认间隔 `--recheck-s`（默认 5 s）。
- 看不到目标时原地左转 30° 再找，转满一圈还找不到就停。
- **没有避障**，只能在空地上用，人要在旁边随时 Ctrl-C。目标必须站在地面上，桌面上的东西会算错距离。

## 注意事项

- **对焦**：标定脚本每次启动都会把两台 C920 锁定为固定对焦 `focus_absolute=0`。C920 拔插或重启后会恢复自动对焦，而内参只在标定时的对焦下有效，所以录数据、做导航时也要先锁定对焦：

  ```bash
  for d in /dev/cam_front_left /dev/cam_front_right; do
      v4l2-ctl -d $d -c focus_automatic_continuous=0 -c focus_absolute=0
  done
  ```

- **按键**：这台机器的 OpenCV 用 Qt 显示窗口，只有图像窗口获得焦点时才收得到按键。
- **旧数据集**：`../datasets/bimanual_4cam_test` 是在左右改名前录的，里面的 `front_left` 实际是机器人右侧的相机，`front_right` 是左侧的。和新数据混在一起训练前，要先把这两路对调。
- **腕部相机左右**：`cam_wrist_left` 是否真的装在左臂（`/dev/arm_B_follower`）上还没有验证。可以用 `wrist_camera_test.py` 动一下手臂确认。

## 保存和恢复手臂姿势 `save_pose.py`

只动两只从臂：左臂 `/dev/arm_B_follower`，右臂 `/dev/arm_A_follower`。名字自定，保存和还原用同一个。

```bash
python save_pose.py save home       # 保存当前姿势，并打印每个舵机的角度
python save_pose.py restore home    # 回到这个姿势，并打印每个舵机的角度
```

`save` 不改力矩。`restore` 到位后力矩保持打开，要松开就跑 `python release_arms.py`。
