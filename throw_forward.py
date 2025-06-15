import sys


from lerobot.common.robots.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower import SO101FollowerConfig
import time


def main(robot, pull_back, flick_forward, gripper_open):
    current_obs = robot.get_observation()


    robot.send_action(pull_back)
    time.sleep(1.0)

    robot.send_action(flick_forward)
    time.sleep(0.2)

    #robot.send_action(gripper_open)
    #time.sleep(0.5)

    robot.disconnect()


if __name__ == "__main__":
    PORT = "/dev/tty.usbmodem5A7A0186431"
    ROBOT_ID = "my_awesome_follower_arm"

    PULL_BACK = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 10,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -10.0,
        "gripper.pos": 10.0,
    }

    _PULL_BACK = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 45.0,
        "elbow_flex.pos": 90.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -30.0,
        "gripper.pos": 20.0,
    }
    
    FLICK_FORWARD = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": -90,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -10.0,
        "gripper.pos": 10.0,
    }

    TEST_FORWARD = {'shoulder_pan.pos': -5.054945054945055,
                     'shoulder_lift.pos': -17.0989010989011,
                       'elbow_flex.pos': -96.04395604395604,
                         'wrist_flex.pos': -13.098901098901099,
                           'wrist_roll.pos': -10.241758241758241,
                             'gripper.pos': 10.142118863049095}
    
    _FLICK_FORWARD = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 70.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -10.0,
        "gripper.pos": 20.0,
    }
    RELEASE_GRIP = {"gripper.pos": 80.0}

    config = SO101FollowerConfig(port=PORT, use_degrees=True, id=ROBOT_ID)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)

    main(robot, PULL_BACK, TEST_FORWARD, RELEASE_GRIP)



