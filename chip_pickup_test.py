from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.robots.so101_follower import SO101FollowerConfig
import numpy as np
import time
import os

# Load your SO-101 robot configuration
def control_so101_programmatically():
    # Create robot config for SO-101
    robot_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem5A7A0186301",  # Your robot's port
        disable_torque_on_disconnect=True,
        max_relative_target=None,
        use_degrees=False
    )
    
    # Initialize robot
    robot = make_robot_from_config(robot_config)
    
    # Connect without calibration since we already have it
    robot.connect(calibrate=False)
    
    try:
        # Get current robot state
        observation = robot.get_observation()
        print(f"Current robot state: {observation}")
        
        # Create custom actions (6 joints + gripper for SO-101)
        # Format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        
        # Example 1: Move to a specific pose
        target_action = {
            "shoulder_pan.pos": 0.1,
            "shoulder_lift.pos": -0.2,
            "elbow_flex.pos": 0.3,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.1,
            "gripper.pos": 1.0  # Open gripper
        }
        robot.send_action(target_action)
        time.sleep(2)
        
        # Example 2: Incremental movements
        for step in range(10):
            current_obs = robot.get_observation()
            
            # Modify action incrementally
            new_action = {
                "shoulder_pan.pos": current_obs["shoulder_pan.pos"] + 0.01,
                "shoulder_lift.pos": current_obs["shoulder_lift.pos"],
                "elbow_flex.pos": current_obs["elbow_flex.pos"],
                "wrist_flex.pos": current_obs["wrist_flex.pos"],
                "wrist_roll.pos": current_obs["wrist_roll.pos"],
                "gripper.pos": current_obs["gripper.pos"]
            }
            
            robot.send_action(new_action)
            time.sleep(0.1)
        
        # Example 3: Control gripper
        gripper_open = {
            "shoulder_pan.pos": current_obs["shoulder_pan.pos"],
            "shoulder_lift.pos": current_obs["shoulder_lift.pos"],
            "elbow_flex.pos": current_obs["elbow_flex.pos"],
            "wrist_flex.pos": current_obs["wrist_flex.pos"],
            "wrist_roll.pos": current_obs["wrist_roll.pos"],
            "gripper.pos": 0.0  # Open gripper
        }
        robot.send_action(gripper_open)
        time.sleep(1)
        
        gripper_close = {
            "shoulder_pan.pos": current_obs["shoulder_pan.pos"],
            "shoulder_lift.pos": current_obs["shoulder_lift.pos"],
            "elbow_flex.pos": current_obs["elbow_flex.pos"],
            "wrist_flex.pos": current_obs["wrist_flex.pos"],
            "wrist_roll.pos": current_obs["wrist_roll.pos"],
            "gripper.pos": 1.0  # Close gripper
        }
        robot.send_action(gripper_close)
        
    finally:
        robot.disconnect()

if __name__ == "__main__":
    control_so101_programmatically()