import sys
import logging  
import time
from typing import Dict, Any, Optional
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


from lerobot.common.robots.so101_follower import SO101Follower
from lerobot.common.robots.so101_follower import SO101FollowerConfig
from lerobot.common.motors.feetech import OperatingMode, TorqueMode

from vibe_code import add_throwing_capability

import time
import numpy as np 

VECTOR_TO_KEY = {1: 'shoulder_pan.pos', 2: 'shoulder_lift.pos', 3: 'elbow_flex.pos', 4: 'wrist_flex.pos', 5: "wrist_roll.pos", 6: "gripper.pos"}

def velocity_move(robot):
    bus = robot.bus 
    for m in bus.motors:
        bus.write("Operating_Mode", m, 1)
        bus.write("Goal_Velocity", m, 400)
    time.sleep(1) #time.sleep(0.12)
    bus.write("Goal_Velocity", m, 0)


class SO101VelocityFlicker:
    """
    Velocity-based flick controller for SO101Follower.
    Provides fast, controlled flicking motions using velocity mode.
    """
    
    def __init__(self, robot):
        """
        Initialize the velocity flick controller.
        
        Args:
            robot: SO101Follower instance
        """
        self.robot = robot
        self.original_operating_modes = {}
        self.flick_in_progress = False
        self.configured_for_flick = False
        self.move_home()
    
    def setup_velocity_flick(self, flick_joint: str = "wrist_flex") -> None:
        """
        Configure the robot for velocity-based flicking.
        
        Args:
            flick_joint: Joint to use for flicking (default: wrist_flex)
        """
        if not self.robot.is_connected:
            raise Exception("Robot must be connected before setup")
        
        
        if self.configured_for_flick:
            logger.warning("Already configured for flicking")
            return
        
        logger.info(f"Configuring {flick_joint} for velocity-based flicking...")
        
        # Store original operating modes for all motors
        for motor in self.robot.bus.motors:
            current_mode = self.robot.bus.read("Operating_Mode", motor)
            self.original_operating_modes[motor] = current_mode
        
        # Disable torque to change operating mode
        self.robot.bus.disable_torque()
        
        # Set flick joint to velocity mode
        self.robot.bus.write("Operating_Mode", flick_joint, OperatingMode.VELOCITY.value)
        
        # Keep other joints in position mode for stability
        for motor in self.robot.bus.motors:
            if motor != flick_joint:
                self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        
        # Configure velocity parameters for smoother control
        # Set acceleration to maximum for snappy response
        self.robot.bus.write("Acceleration", flick_joint, 254)
        self.robot.bus.write("Maximum_Acceleration", flick_joint, 254)
        
        # Re-enable torque
        self.robot.bus.enable_torque()
        
        self.configured_for_flick = True
        logger.info(f"Successfully configured {flick_joint} for velocity flicking") 
    
    def quick_flick(self, 
                    flick_joint: str = "wrist_flex",
                    flick_speed: int = 1000,
                    flick_duration: float = 0.08) -> None:
        """
        Perform a quick single-direction flick.
        
        Args:
            flick_joint: Joint to flick with
            flick_speed: Velocity for flick motion (0-2048, higher = faster)
            flick_duration: How long to apply velocity (seconds)
        """
        if not self.configured_for_flick:
            self.setup_velocity_flick(flick_joint)
        
        if self.flick_in_progress:
            logger.warning("Flick already in progress, ignoring request")
            return
        
        self._execute_velocity_sequence(
            flick_joint=flick_joint,
            sequence=[(flick_speed, flick_duration), (0, 0.1)],
            description="quick flick"
        )

    def _execute_velocity_sequence(self,
                                  flick_joint: str,
                                  sequence: list[tuple[int, float]],
                                  description: str) -> None:
        """
        Execute a sequence of velocity commands.
        
        Args:
            flick_joint: Joint to control
            sequence: List of (velocity, duration) tuples
            description: Description for logging
        """
        self.flick_in_progress = True
        
        try:
            logger.info(f"Starting {description} on {flick_joint}")
            
            for velocity, duration in sequence:
                if duration > 0:
                    #current_vel = self.robot.bus.read("Present_Velocity", flick_joint)
                    #logger.info(f"Current velocity: {current_vel}, Setting: {velocity}")

                    self.robot.bus.write("Goal_Velocity", flick_joint, velocity)

                    goal_vel = self.robot.bus.read("Goal_Velocity", flick_joint)
                    logger.info(f"Goal velocity set to: {goal_vel}")
                    time.sleep(duration)

                    #actual_vel = self.robot.bus.read("Present_Velocity", flick_joint)
                    #logger.info(f"Actual velocity during motion: {actual_vel}")
            
            # Ensure motor stops at the end
            self.robot.bus.write("Goal_Velocity", flick_joint, 0)
            
            logger.info(f"Completed {description}")
            
        except Exception as e:
            logger.error(f"Error during {description}: {e}")
            # Emergency stop
            self.robot.bus.write("Goal_Velocity", flick_joint, 0)
            raise
        
        finally:
            self.flick_in_progress = False

    def flick_pwm_mode(self,
                       flick_joint: str = "wrist_flex", 
                       flick_pwm: int = 800,
                       flick_duration: float = 0.05,
                       return_pwm: int = -400,
                       return_duration: float = 0.15) -> None:
        """
        Perform a flick using PWM control mode (most aggressive).
        
        Args:
            flick_joint: Joint to flick with
            flick_pwm: PWM value for flick (higher = more aggressive)
            flick_duration: How long to apply flick PWM (seconds)
            return_pwm: PWM for return motion (negative for opposite direction)
            return_duration: How long to apply return PWM (seconds)
        """
        if self.flick_in_progress:
            logger.warning("Flick already in progress, ignoring request")
            return
        
        self.flick_in_progress = True
        
        try:
            # Switch to PWM mode temporarily
            self.robot.bus.disable_torque([flick_joint])
            self.robot.bus.write("Operating_Mode", flick_joint, OperatingMode.PWM.value)
            self.robot.bus.enable_torque([flick_joint])
            
            logger.info(f"Starting PWM flick motion on {flick_joint}")
            
            # Apply flick PWM
            self.robot.bus.write("Goal_PWM", flick_joint, flick_pwm)
            time.sleep(flick_duration)
            
            # Apply return PWM
            self.robot.bus.write("Goal_PWM", flick_joint, return_pwm)
            time.sleep(return_duration)
            
            # Stop motion
            self.robot.bus.write("Goal_PWM", flick_joint, 0)
            
            logger.info("PWM flick motion completed")
            
        except Exception as e:
            logger.error(f"Error during PWM flick motion: {e}")
            # Emergency stop
            self.robot.bus.write("Goal_PWM", flick_joint, 0)
        
        finally:
            self.flick_in_progress = False
    
    def stop_flick(self, flick_joint: str = "wrist_flex") -> None:
        """
        Emergency stop for the flick joint.
        
        Args:
            flick_joint: Joint to stop
        """
        self.robot.bus.write("Goal_Velocity", flick_joint, 0)
        self.flick_in_progress = False
        logger.info(f"Emergency stop applied to {flick_joint}")

    def move_home(self):
        print("Moving home")
        HOME = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -0.0,
        "gripper.pos": 0.0,
        }

        self.robot.send_action(HOME)
        time.sleep(1)
    


    def restore_position_mode(self) -> None:
        """
        Restore all motors to their original operating modes.
        """
        if not self.original_operating_modes:
            logger.warning("No original operating modes stored")
            return
        
        logger.info("Restoring original operating modes...")
        
        # Stop any ongoing motion first
        for motor in self.robot.bus.motors:
            try:
                self.robot.bus.write("Goal_Velocity", motor, 0)
            except:
                pass  # Motor might not be in velocity mode
        
        # Disable torque to change modes
        self.robot.bus.disable_torque()
        
        # Restore original modes
        for motor, mode in self.original_operating_modes.items():
            self.robot.bus.write("Operating_Mode", motor, mode)
        
        # Re-enable torque
        self.robot.bus.enable_torque()
        
        # Clear stored states
        self.original_operating_modes.clear()
        self.configured_for_flick = False
        self.flick_in_progress = False
        
        logger.info("Successfully restored to original operating modes")




def main(robot, pull_back, flick_forward, gripper_open, duration: float, n_points=10):
    '''
    Moves from pull_back position to flick forward position in duration (s)
    '''
    current_obs = robot.get_observation()
    # move to starting state 
    robot.send_action(pull_back)
    #time.sleep(1)
    
    # forward trajectory 
    start_vector = np.array([pull_back[VECTOR_TO_KEY[i]] for i in range(1,7)])
    end_vector = np.array([flick_forward[VECTOR_TO_KEY[i]] for i in range(1,7)])

    all_inter_states = np.linspace(start_vector, end_vector, num=n_points, endpoint=True)
    for j in range(0, len(all_inter_states)):
        inter_state = {VECTOR_TO_KEY[i]: all_inter_states[j][i-1] for i in range(1,7) }

        robot.send_action(inter_state)
        time.sleep(0.0001)
        #time.sleep(duration / n_points)
    
    for i in range(5):
        robot.send_action(flick_forward)
        time.sleep(0.1)


    #robot.send_action(flick_forward)
    #time.sleep(0.2)

    #robot.send_action(gripper_open)
    #time.sleep(0.5)

    robot.disconnect()


def driver():
    PORT = "/dev/tty.usbmodem5A7A0186401"
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
    
    HOME = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": -0.0,
        "gripper.pos": 0.0,
    }
    RELEASE_GRIP = {"gripper.pos": 10.0}


    config = SO101FollowerConfig(port=PORT, use_degrees=True, id=ROBOT_ID)
    robot = SO101Follower(config)
    # change the operating mode
    robot.connect(calibrate=False)

    add_throwing_capability(robot)

    robot.throw_fast(strength=0.7)
   
    robot.disconnect()


    
    
    
    
    return 
    robot.bus.enable_torque()

    for m in robot.bus.motors:
        robot.bus.write("Return_Delay_Time", m, 0, normalize=False)
        robot.bus.write("Maximum_Acceleration", m, 254, normalize=False)
        robot.bus.write("Acceleration", m, 254, normalize=False)

    #robot.send_action(HOME)
    #time.sleep(0.001)
    #robot.send_action(RELEASE_GRIP)
    #robot.send_action(TEST_FORWARD)

    #time.sleep(1)

    #robot.disconnect()

    robot.send_action(HOME)

    flicker = SO101VelocityFlicker(robot)

    flicker.flick_pwm_mode(
            flick_joint="wrist_flex",
            flick_pwm=900,
            flick_duration=0.05,
            return_pwm=-450,
            return_duration=0.12
    )


    return 

    flicker = SO101VelocityFlicker(robot)
    print("1. Quick flick...")
    flicker.quick_flick(
             flick_joint="wrist_flex",
             flick_speed=-1000,
             flick_duration=2
    )
    flicker.restore_position_mode()
    
    robot.disconnect()

    
    
    
    """
    
    

    with robot.bus.torque_disabled():
        for m in robot.bus.motors:
            robot.bus.write("Operating_Mode", m, 1)      # velocity mode
            robot.bus.write("Goal_Velocity", m, 0)

    

    #main(robot, PULL_BACK, TEST_FORWARD, RELEASE_GRIP, duration=1)
    velocity_move(robot)
    """



if __name__ == "__main__":
    driver()
    pass 



