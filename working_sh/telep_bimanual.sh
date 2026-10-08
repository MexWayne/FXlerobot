lerobot-teleoperate \
    --robot.type=bi_so_follower \
    --robot.id=bimanual_follower \
    --robot.left_arm_config.port=/dev/arm_B_follower \
    --robot.right_arm_config.port=/dev/arm_A_follower \
    --teleop.type=bi_so_leader \
    --teleop.id=bimanual_leader \
    --teleop.left_arm_config.port=/dev/arm_B_leader \
    --teleop.right_arm_config.port=/dev/arm_A_leader
