REPO_ID=${REPO_ID:-mex_wayne/case_vacuum_under_the_chairs_00009}
TASK=${TASK:-"vacuum under the chairs."}
NUM_EPISODES=${NUM_EPISODES:-5}
DATASET_ROOT=${DATASET_ROOT:-$HOME/workspaces/FXLerobot/FXlerobot/datasets/${REPO_ID##*/}}
mkdir -p "$(dirname "$DATASET_ROOT")"

DISPLAY_DATA=${DISPLAY_DATA:-true}
if [ -z "$DISPLAY" ] && [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
    [ -z "$XAUTHORITY" ] && [ -f /run/user/$(id -u)/gdm/Xauthority ] && export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
fi
if [ -z "$DISPLAY" ] || ! xset q >/dev/null 2>&1; then
    echo "No usable display, recording without rerun viewer"
    DISPLAY_DATA=false
fi

# Over SSH, DISPLAY points at the robot's own screen, so pynput would listen to the robot's
# keyboard; read n/r/q from this SSH terminal instead.
if [ -n "$SSH_CONNECTION" ] && [ -t 0 ]; then
    export LEROBOT_KEYBOARD_BACKEND=terminal
    echo "Started over SSH: n/r/q keys are read from this terminal; rerun stays on the robot screen"
fi

# Frames go straight into h264 encoders instead of PNG files: writing 120 PNG/s starves the
# control loop (follower turns jerky). libsvtav1 can't keep up with 4 streams here and
# h264_nvenc doesn't exist on Jetson; h264 veryfast uses ~2.7 cores with no dropped frames.

# Names come from 99-lerobot-camera.rules (robot's own left/right).
CAM_FRONT_LEFT=/dev/cam_front_left
CAM_FRONT_RIGHT=/dev/cam_front_right
CAM_WRIST_LEFT=/dev/cam_wrist_left
CAM_WRIST_RIGHT=/dev/cam_wrist_right

# Wrist cams drop to ~8-15 fps in low light unless this is off.
for cam in "$CAM_WRIST_LEFT" "$CAM_WRIST_RIGHT"; do
    if [ -e "$cam" ]; then
        v4l2-ctl -d "$(readlink -f "$cam")" -c exposure_dynamic_framerate=0 >/dev/null 2>&1 || true
    else
        echo "WARNING: missing camera path: $cam" >&2
    fi
done

lerobot-record \
    --robot.type=bi_so_follower \
    --robot.id=bimanual_follower \
    --robot.left_arm_config.port=/dev/arm_B_follower \
    --robot.right_arm_config.port=/dev/arm_A_follower \
    --robot.cameras="{ front_left: {type: opencv, index_or_path: \"$CAM_FRONT_LEFT\", width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, front_right: {type: opencv, index_or_path: \"$CAM_FRONT_RIGHT\", width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, wrist_left: {type: opencv, index_or_path: \"$CAM_WRIST_LEFT\", width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, wrist_right: {type: opencv, index_or_path: \"$CAM_WRIST_RIGHT\", width: 640, height: 480, fps: 30, fourcc: \"MJPG\"} }" \
    --teleop.type=bi_so_leader \
    --teleop.id=bimanual_leader \
    --teleop.left_arm_config.port=/dev/arm_B_leader \
    --teleop.right_arm_config.port=/dev/arm_A_leader \
    --display_data="$DISPLAY_DATA" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.rgb_encoder.vcodec=h264 \
    --dataset.rgb_encoder.preset=veryfast \
    --dataset.root="$DATASET_ROOT"
