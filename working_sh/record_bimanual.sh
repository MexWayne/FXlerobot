REPO_ID=${REPO_ID:-mex_wayne/bimanual_4cam_test}
TASK=${TASK:-"Grab the black cube"}
NUM_EPISODES=${NUM_EPISODES:-5}
DATASET_ROOT=${DATASET_ROOT:-$HOME/workspaces/FXLerobot/datasets/${REPO_ID##*/}}
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
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
    --dataset.root="$DATASET_ROOT"
