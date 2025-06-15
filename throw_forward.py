import sys


from lerobot.common.robots.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower import SO101FollowerConfig
import time
import numpy as np 

VECTOR_TO_KEY = {1: 'shoulder_pan.pos', 2: 'shoulder_lift.pos', 3: 'elbow_flex.pos', 4: 'wrist_flex.pos', 5: "wrist_roll.pos", 6: "gripper.pos"}


def main(robot, pull_back, flick_forward, gripper_open, duration: float, n_points=10):
    '''
    Moves from pull_back position to flick forward position in duration (s)
    '''
    current_obs = robot.get_observation()
    # move to starting state 
    robot.send_action(pull_back)
    time.sleep(1)
    
    # forward trajectory 
    start_vector = np.array([pull_back[VECTOR_TO_KEY[i]] for i in range(1,7)])
    end_vector = np.array([flick_forward[VECTOR_TO_KEY[i]] for i in range(1,7)])

    all_inter_states = np.linspace(start_vector, end_vector, num=n_points, endpoint=True)
    for j in range(0, len(all_inter_states)):
        inter_state = {VECTOR_TO_KEY[i]: all_inter_states[j][i-1] for i in range(1,7) }

        robot.send_action(inter_state)
        time.sleep(duration / n_points)
    
    for i in range(5):
        robot.send_action(flick_forward)
        time.sleep(0.1)


    #robot.send_action(flick_forward)
    #time.sleep(0.2)

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

    main(robot, PULL_BACK, TEST_FORWARD, RELEASE_GRIP, duration=1)



