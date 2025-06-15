#!/usr/bin/env python
# Fast throwing implementation for SO101 Follower Robot

import time
import logging
from typing import Dict, Any
from lerobot.common.motors.feetech import OperatingMode

logger = logging.getLogger(__name__)

class SO101ThrowingController:
    """
    Simple throwing controller for SO101 Follower robot.
    Implements fast throwing motions by maximizing motor speeds and accelerations.
    """
    
    def __init__(self, robot):
        self.robot = robot
        self.original_coefficients = {}
        
    def setup_for_throwing(self):
        """Configure motors for maximum speed throwing."""
        logger.info("Setting up robot for fast throwing...")
        
        # Store original PID values for restoration later
        for motor in self.robot.bus.motors:
            self.original_coefficients[motor] = {
                'P': self.robot.bus.read("P_Coefficient", motor, normalize=False),
                'I': self.robot.bus.read("I_Coefficient", motor, normalize=False), 
                'D': self.robot.bus.read("D_Coefficient", motor, normalize=False),
                'accel': self.robot.bus.read("Maximum_Acceleration", motor, normalize=False)
                # Note: Maximum_Velocity doesn't exist in STS3215 control table
            }
        
        with self.robot.bus.torque_disabled():
            for motor in self.robot.bus.motors:
                # Set to position mode for precise control
                self.robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                
                # More conservative settings to avoid overload protection
                self.robot.bus.write("Maximum_Acceleration", motor, 200, normalize=False)  # Reduced from 254
                self.robot.bus.write("Acceleration", motor, 200, normalize=False)
                
                # More moderate P gain to avoid oscillation and overload
                self.robot.bus.write("P_Coefficient", motor, 48, normalize=False)  # Reduced from 64
                
                # Reduce I and D to minimize delay
                self.robot.bus.write("I_Coefficient", motor, 0, normalize=False)
                self.robot.bus.write("D_Coefficient", motor, 8, normalize=False)  # Reduce from 32
        
        logger.info("Robot configured for throwing")
    
    def restore_original_settings(self, retry_attempts=3):
        """Restore original motor settings with error recovery."""
        logger.info("Restoring original motor settings...")
        
        # Wait a bit for motors to cool down/recover
        time.sleep(1.0)
        
        for attempt in range(retry_attempts):
            try:
                with self.robot.bus.torque_disabled():
                    for motor in self.robot.bus.motors:
                        if motor in self.original_coefficients:
                            coeffs = self.original_coefficients[motor]
                            try:
                                self.robot.bus.write("P_Coefficient", motor, coeffs['P'], normalize=False)
                                self.robot.bus.write("I_Coefficient", motor, coeffs['I'], normalize=False)
                                self.robot.bus.write("D_Coefficient", motor, coeffs['D'], normalize=False)
                                self.robot.bus.write("Maximum_Acceleration", motor, coeffs['accel'], normalize=False)
                            except Exception as e:
                                logger.warning(f"Failed to restore settings for motor {motor}: {e}")
                                continue
                logger.info("Settings restored successfully")
                return
            except Exception as e:
                logger.warning(f"Restore attempt {attempt + 1} failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2.0)  # Wait longer between attempts
                else:
                    logger.error("Failed to restore settings after all attempts")
                    # Try to at least enable torque back
                    try:
                        self.robot.bus.enable_torque()
                    except:
                        pass
    
    def throw_overhand(self, throw_strength: float = 1.0, recovery_time: float = 0.5):
        """
        Simple overhand throwing motion with motor protection.
        
        Args:
            throw_strength: Multiplier for throw intensity (0.5-1.0 recommended)
            recovery_time: Time to wait between motion phases for motor recovery
        """
        logger.info(f"Executing overhand throw with strength {throw_strength}")
        
        # Define throwing positions in DEGREES (adjust these based on your robot's workspace)
        # Joint positions are in degrees, gripper is 0-100 (always uses RANGE_0_100 mode)
        wind_up_pos = {
            "shoulder_pan": 0,      # Centered
            "shoulder_lift": -45,   # Shoulder back (degrees)
            "elbow_flex": -60,      # Elbow bent back (degrees)
            "wrist_flex": -30,      # Wrist cocked back (degrees)
            "wrist_roll": 0,        # Neutral
            "gripper": 100          # Gripping object (0-100 range)
        }
        
        throw_pos = {
            "shoulder_pan": 0,      # Still centered
            "shoulder_lift": 30,    # Shoulder forward (degrees)
            "elbow_flex": 20,       # Elbow extending (degrees)
            "wrist_flex": 25,       # Wrist snapping forward (degrees)
            "wrist_roll": 0,        # Neutral
            "gripper": 100          # Still gripping (0-100 range)
        }
        
        release_pos = {
            "shoulder_pan": 0,
            "shoulder_lift": 45,    # Follow through (degrees)
            "elbow_flex": 40,       # Full extension (degrees)
            "wrist_flex": 40,       # Full snap (degrees)
            "wrist_roll": 0,
            "gripper": 0            # Release! (0-100 range)
        }
        
        # Scale positions by throw strength
        for pos_dict in [wind_up_pos, throw_pos, release_pos]:
            for joint in pos_dict:
                if joint != "gripper":  # Don't scale gripper
                    pos_dict[joint] *= throw_strength
        
        try:
            # Step 1: Wind up (slower)
            action1 = {f"{joint}.pos": pos for joint, pos in wind_up_pos.items()}
            self.robot.send_action(action1)
            time.sleep(1.0 + recovery_time)  # Allow time to reach position and motors to settle
            
            # Step 2: Throw motion (fast!)
            action2 = {f"{joint}.pos": pos for joint, pos in throw_pos.items()}
            self.robot.send_action(action2)
            time.sleep(0.2 + recovery_time)  # Quick transition but allow motor recovery
            
            # Step 3: Release and follow through
            action3 = {f"{joint}.pos": pos for joint, pos in release_pos.items()}
            self.robot.send_action(action3)
            time.sleep(0.5 + recovery_time)
            
            # Give motors time to cool down
            time.sleep(2.0)
            
            logger.info("Throw completed!")
            
        except Exception as e:
            logger.error(f"Error during throw: {e}")
            # Wait for motors to recover before raising
            time.sleep(3.0)
            raise
    
    def throw_sidearm(self, throw_strength: float = 1.0):
        """
        Sidearm/baseball style throw.
        
        Args:
            throw_strength: Multiplier for throw intensity
        """
        logger.info(f"Executing sidearm throw with strength {throw_strength}")
        
        wind_up_pos = {
            "shoulder_pan": -40 * throw_strength,   # Rotate away (degrees)
            "shoulder_lift": 0,                     # Level
            "elbow_flex": -30,                      # Elbow back (degrees)
            "wrist_flex": -20,                      # Wrist back (degrees)
            "wrist_roll": 0,
            "gripper": 100
        }
        
        throw_pos = {
            "shoulder_pan": 40 * throw_strength,    # Rotate through (degrees)
            "shoulder_lift": 10,                    # Slight up (degrees)
            "elbow_flex": 10,                       # Extending (degrees)
            "wrist_flex": 20,                       # Snap forward (degrees)
            "wrist_roll": 0,
            "gripper": 100
        }
        
        release_pos = {
            "shoulder_pan": 50 * throw_strength,    # Full rotation (degrees)
            "shoulder_lift": 20,                    # Follow up (degrees)
            "elbow_flex": 30,                       # Extended (degrees)
            "wrist_flex": 30,                       # Snapped (degrees)
            "wrist_roll": 0,
            "gripper": 0                            # Release!
        }
        
        try:
            # Wind up
            action1 = {f"{joint}.pos": pos for joint, pos in wind_up_pos.items()}
            self.robot.send_action(action1)
            time.sleep(0.6)
            
            # Throw 
            action2 = {f"{joint}.pos": pos for joint, pos in throw_pos.items()}
            self.robot.send_action(action2)
            time.sleep(0.1)  # Very fast transition
            
            # Release
            action3 = {f"{joint}.pos": pos for joint, pos in release_pos.items()}
            self.robot.send_action(action3)
            time.sleep(0.2)
            
            logger.info("Sidearm throw completed!")
            
        except Exception as e:
            logger.error(f"Error during sidearm throw: {e}")
            raise
    
    def rapid_fire_throw(self, num_throws: int = 3, delay: float = 0.5):
        """
        Multiple quick throws in succession.
        
        Args:
            num_throws: Number of throws to execute
            delay: Delay between throws in seconds
        """
        logger.info(f"Executing {num_throws} rapid throws")
        
        for i in range(num_throws):
            logger.info(f"Throw {i+1}/{num_throws}")
            self.throw_overhand(throw_strength=0.8)  # Slightly reduced strength for speed
            if i < num_throws - 1:  # Don't delay after last throw
                time.sleep(delay)


# Example usage function
def demo_throwing(robot):
    """
    Demonstration of different throwing techniques.
    
    Args:
        robot: Connected SO101Follower instance
    """
    if not robot.is_connected:
        logger.error("Robot not connected!")
        return
    
    controller = SO101ThrowingController(robot)
    
    try:
        # Setup for throwing
        controller.setup_for_throwing()
        
        print("Robot ready for throwing!")
        print("Available commands:")
        print("1. Overhand throw")
        print("2. Sidearm throw") 
        print("3. Rapid fire (3 throws)")
        print("4. Custom strength overhand")
        print("q. Quit")
        
        while True:
            choice = input("\nEnter choice: ").strip().lower()
            
            if choice == '1':
                controller.throw_overhand()
            elif choice == '2':
                controller.throw_sidearm()
            elif choice == '3':
                controller.rapid_fire_throw()
            elif choice == '4':
                strength = float(input("Enter throw strength (0.1-1.0): "))
                controller.throw_overhand(throw_strength=strength)
            elif choice == 'q':
                break
            else:
                print("Invalid choice!")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    finally:
        # Always restore original settings
        controller.restore_original_settings()
        logger.info("Demo completed")


# Quick setup function to add to your SO101Follower class
def add_throwing_capability(robot):
    """
    Add throwing methods directly to your robot instance.
    
    Usage:
        robot = SO101Follower(config)
        robot.connect()
        add_throwing_capability(robot)
        robot.throw_fast()
    """
    controller = SO101ThrowingController(robot)
    
    def throw_fast(strength=1.0):
        controller.setup_for_throwing()
        try:
            controller.throw_overhand(strength)
        finally:
            controller.restore_original_settings()
    
    def throw_sidearm_fast(strength=1.0):
        controller.setup_for_throwing()
        try:
            controller.throw_sidearm(strength)
        finally:
            controller.restore_original_settings()
    
    # Add methods to robot instance
    robot.throw_fast = throw_fast
    robot.throw_sidearm_fast = throw_sidearm_fast
    robot.throwing_controller = controller
    
    return robot


if __name__ == "__main__":
    # Example of how to use with your robot
    print("Throwing controller for SO101 Follower")
    print("Import this module and use add_throwing_capability(robot) or demo_throwing(robot)")