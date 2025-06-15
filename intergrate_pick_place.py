#!/usr/bin/env python3
"""
Real LeRobot Hardware Integration with SmolVLM for vision-based control
Works with actual LeRobot hardware (Koch, ALOHA, SO100, SO101, etc.)
"""

import numpy as np
import torch
import cv2
from PIL import Image
import re
import time
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# LeRobot imports
from lerobot.common.robots import Robot, make_robot_from_config
from lerobot.common.robots.koch_follower.config_koch_follower import KochFollowerConfig
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.cameras import Camera, CameraConfig, make_cameras_from_configs
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_logging

# Try to import teleoperator classes if available
try:
    from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
    from lerobot.common.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.common.teleoperators.utils import make_teleoperator_from_config
    TELEOP_AVAILABLE = True
except ImportError:
    print("Warning: Teleoperator modules not found. Teleoperation will be disabled.")
    TELEOP_AVAILABLE = False

# Vision model imports
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoProcessor, VisionEncoderDecoderModel as AutoModelForImageTextToText
        print("Using VisionEncoderDecoderModel as fallback")

print("Loading LeRobot Hardware Interface with SmolVLM...")

@dataclass
class PickPlaceController:
    """Controller for autonomous pick and place operations"""
    
    def __init__(self, robot_instance):
        self.robot = robot_instance
        self.pick_height = 50  # mm above table
        self.approach_height = 150  # mm for approach
        self.drop_location = (200, 300)  # Default drop location in mm
        self.gripper_open_pos = 80.0
        self.gripper_close_pos = 20.0
        
        # State machine
        self.state = "SEARCHING"  # SEARCHING, APPROACHING, GRASPING, LIFTING, MOVING, DROPPING, DONE
        self.target_object = None
        self.last_detection_time = 0
        
    def detect_objects_with_depth_only(self, image, depth_frame):
        """Detect objects using depth information when vision model is not available"""
        detections = {
            'objects': [],
            'targets': [],
            'actions': [],
            'robot_coords': []
        }
        
        if depth_frame is None:
            return detections
            
        try:
            # Threshold depth to find objects on table
            # Assuming table is at ~500-700mm depth
            table_depth_min = 400
            table_depth_max = 800
            
            # Create mask for objects on table
            mask = np.where((depth_frame > table_depth_min) & (depth_frame < table_depth_max), 255, 0).astype(np.uint8)
            
            # Remove noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum object size
                    valid_contours.append(contour)
            
            # Sort by area (largest first)
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            
            # Process top 3 objects
            for i, contour in enumerate(valid_contours[:3]):
                # Get center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get depth at center
                    depth_at_center = depth_frame[cy, cx]
                    
                    # Add to detections
                    detections['objects'].append((cx, cy))
                    
                    # Convert to robot coordinates if calibrated
                    if self.robot.camera_calibration is not None:
                        robot_x, robot_y = self.robot.pixel_to_robot(cx, cy)
                        detections['robot_coords'].append((robot_x, robot_y, depth_at_center))
                        
                    # Draw on image
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(image, f"Obj{i+1} D:{int(depth_at_center)}mm", (cx-40, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Determine action based on state
            if self.state == "SEARCHING" and detections['objects']:
                detections['actions'].append('move_to_object')
            
            return detections
            
        except Exception as e:
            print(f"Depth detection error: {e}")
            return detections
    
    def inverse_kinematics_so101(self, x, y, z, gripper_angle=0):
        """
        Simple inverse kinematics for SO101 robot
        Returns joint angles to reach target position
        """
        # Robot dimensions (approximate - adjust for your robot)
        L1 = 100  # Base to shoulder height
        L2 = 150  # Upper arm length  
        L3 = 150  # Forearm length
        L4 = 100  # Wrist to gripper length
        
        # Calculate distance to target
        dist_xy = np.sqrt(x**2 + y**2)
        
        # Shoulder pan angle
        shoulder_pan = np.arctan2(x, y) * 180 / np.pi
        
        # Effective reach in z-plane
        z_eff = z - L1
        reach = np.sqrt(dist_xy**2 + z_eff**2)
        
        # Check if reachable
        max_reach = L2 + L3 + L4
        if reach > max_reach * 0.95:  # 95% of max reach for safety
            print(f"Target out of reach: {reach}mm > {max_reach}mm")
            return None
            
        # Calculate elbow angle using law of cosines
        cos_elbow = (L2**2 + L3**2 - reach**2) / (2 * L2 * L3)
        cos_elbow = np.clip(cos_elbow, -1, 1)
        elbow_angle = np.arccos(cos_elbow) * 180 / np.pi - 180
        
        # Calculate shoulder lift angle
        alpha = np.arctan2(z_eff, dist_xy) * 180 / np.pi
        cos_beta = (L2**2 + reach**2 - L3**2) / (2 * L2 * reach)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta) * 180 / np.pi
        shoulder_lift = -(alpha + beta)
        
        # Wrist angle to keep gripper level
        wrist_flex = -(shoulder_lift + elbow_angle + gripper_angle)
        
        return {
            'shoulder_pan.pos': np.clip(shoulder_pan, -90, 90),
            'shoulder_lift.pos': np.clip(shoulder_lift, -90, 0),
            'elbow_flex.pos': np.clip(elbow_angle, -135, 0),
            'wrist_flex.pos': np.clip(wrist_flex, -90, 90),
            'wrist_roll.pos': 0.0
        }
    
    def execute_pick_place(self, detections):
        """Execute pick and place state machine"""
        current_time = time.time()
        
        # Update target object
        if detections['robot_coords'] and (self.target_object is None or self.state == "SEARCHING"):
            # Pick closest object
            self.target_object = min(detections['robot_coords'], key=lambda p: np.sqrt(p[0]**2 + p[1]**2))
            self.last_detection_time = current_time
            
        # State machine
        if self.state == "SEARCHING":
            if self.target_object:
                print(f"Found object at {self.target_object[:2]}mm")
                self.state = "APPROACHING"
            else:
                # Keep searching - move to scan position
                return {
                    'shoulder_pan.pos': np.sin(current_time) * 30,  # Scan left-right
                    'shoulder_lift.pos': -45.0,
                    'elbow_flex.pos': -45.0,
                    'wrist_flex.pos': -45.0,
                    'wrist_roll.pos': 0.0,
                    'gripper.pos': self.gripper_open_pos
                }
                
        elif self.state == "APPROACHING":
            if self.target_object:
                x, y, depth = self.target_object
                # Move above object
                target_pos = self.inverse_kinematics_so101(x, y, self.approach_height)
                if target_pos:
                    target_pos['gripper.pos'] = self.gripper_open_pos
                    print(f"Approaching object at ({x:.1f}, {y:.1f})mm")
                    
                    # Check if we're close to target position
                    obs = self.robot.robot.get_observation()
                    if self._is_at_position(obs, target_pos, tolerance=10):
                        self.state = "GRASPING"
                        time.sleep(0.5)  # Stabilize
                    
                    return target_pos
                    
        elif self.state == "GRASPING":
            if self.target_object:
                x, y, depth = self.target_object
                # Move down to pick height
                target_pos = self.inverse_kinematics_so101(x, y, self.pick_height)
                if target_pos:
                    target_pos['gripper.pos'] = self.gripper_open_pos
                    print(f"Moving down to grasp at height {self.pick_height}mm")
                    
                    # Check if we're at pick position
                    obs = self.robot.robot.get_observation()
                    if self._is_at_position(obs, target_pos, tolerance=5):
                        # Close gripper
                        target_pos['gripper.pos'] = self.gripper_close_pos
                        print("Closing gripper...")
                        self.state = "LIFTING"
                        time.sleep(1.0)  # Wait for gripper to close
                        
                    return target_pos
                    
        elif self.state == "LIFTING":
            if self.target_object:
                x, y, _ = self.target_object
                # Lift up
                target_pos = self.inverse_kinematics_so101(x, y, self.approach_height)
                if target_pos:
                    target_pos['gripper.pos'] = self.gripper_close_pos
                    print("Lifting object...")
                    
                    obs = self.robot.robot.get_observation()
                    if self._is_at_position(obs, target_pos, tolerance=10):
                        self.state = "MOVING"
                        time.sleep(0.5)
                        
                    return target_pos
                    
        elif self.state == "MOVING":
            # Move to drop location
            drop_x, drop_y = self.drop_location
            target_pos = self.inverse_kinematics_so101(drop_x, drop_y, self.approach_height)
            if target_pos:
                target_pos['gripper.pos'] = self.gripper_close_pos
                print(f"Moving to drop location ({drop_x}, {drop_y})mm")
                
                obs = self.robot.robot.get_observation()
                if self._is_at_position(obs, target_pos, tolerance=10):
                    self.state = "DROPPING"
                    time.sleep(0.5)
                    
                return target_pos
                
        elif self.state == "DROPPING":
            # Lower to drop height
            drop_x, drop_y = self.drop_location
            target_pos = self.inverse_kinematics_so101(drop_x, drop_y, self.pick_height)
            if target_pos:
                # Open gripper
                target_pos['gripper.pos'] = self.gripper_open_pos
                print("Dropping object...")
                
                obs = self.robot.robot.get_observation()
                if self._is_at_position(obs, target_pos, tolerance=5):
                    self.state = "DONE"
                    time.sleep(1.0)
                    
                return target_pos
                
        elif self.state == "DONE":
            # Return to home and reset
            print("Task completed! Returning to home...")
            self.state = "SEARCHING"
            self.target_object = None
            
            return {
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -30.0,
                'elbow_flex.pos': 0.0,
                'wrist_flex.pos': -45.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': self.gripper_open_pos
            }
            
        # Default action
        return None
    
    def _is_at_position(self, observation, target_position, tolerance=5):
        """Check if robot is at target position within tolerance"""
        # This is simplified - you may need to adjust based on your observation format
        for key in ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos']:
            if key in observation and key in target_position:
                current = observation[key]
                target = target_position[key]
                if abs(current - target) > tolerance:
                    return False
        return True


class SimulatedRobotWrapper:
    """Wrapper to make simulation environment compatible with robot interface"""
    
    def __init__(self, env):
        self.env = env
        self.is_connected = True
        self.is_calibrated = True
        
        # Define features based on ALOHA
        self.observation_features = {
            "agent_pos": {"shape": (14,), "dtype": "float32"},  # 7 DOF per arm
            "pixels.top": {"shape": (480, 640, 3), "dtype": "uint8"}
        }
        
        self.action_features = {
            "action": {"shape": (14,), "dtype": "float32"}  # 7 DOF per arm
        }
        
        # Reset to get initial observation
        self.last_obs, _ = self.env.reset()
    
    def connect(self):
        """Already connected in simulation"""
        pass
    
    def disconnect(self):
        """Close simulation"""
        self.env.close()
    
    def calibrate(self):
        """No calibration needed for simulation"""
        pass
    
    def get_observation(self):
        """Get observation from simulation"""
        # Convert gym observation to robot format
        obs_dict = {}
        
        if isinstance(self.last_obs, dict):
            if "agent_pos" in self.last_obs:
                obs_dict["agent_pos"] = self.last_obs["agent_pos"][0]  # Remove batch dim
            if "pixels" in self.last_obs and "top" in self.last_obs["pixels"]:
                obs_dict["pixels.top"] = self.last_obs["pixels"]["top"][0]
        
        return obs_dict
    
    def send_action(self, action):
        """Send action to simulation"""
        # Convert robot action dict to gym format
        if isinstance(action, dict):
            action_array = action.get("action", np.zeros(14))
        else:
            action_array = action
        
        # Add batch dimension
        action_batch = np.expand_dims(action_array, axis=0)
        
        # Step simulation
        self.last_obs, reward, terminated, truncated, info = self.env.step(action_batch)
        
        # Reset if episode ended
        if terminated[0] or truncated[0]:
            self.last_obs, _ = self.env.reset()
        
        return {"action": action_array}


class LeRobotHardwareWithVision:
    """Real LeRobot hardware interface with SmolVLM vision"""
    
    def __init__(self, robot_type="koch", teleop_type=None, calibration_path=None, use_sim=False):
        """
        Initialize robot hardware and vision system
        
        Args:
            robot_type: Type of robot ('koch', 'aloha', 'so100', 'so101')
            teleop_type: Type of teleoperator (e.g., 'so100_leader', 'koch_leader')
            calibration_path: Path to calibration directory
            use_sim: Use simulation instead of real hardware
        """
        self.robot_type = robot_type
        self.use_sim = use_sim
        
        # Initialize logging
        init_logging()
        
        # Create robot configuration
        print(f"Initializing {robot_type} {'simulation' if use_sim else 'robot hardware'}...")
        
        if use_sim and robot_type == "aloha":
            # Use ALOHA simulation
            try:
                from lerobot.common.envs.factory import make_env, EnvConfig
                
                print("Creating ALOHA simulation environment...")
                env_config = EnvConfig(
                    env_name="aloha",
                    task="AlohaInsertion-v0",
                    episode_length=400,
                    video_dir=Path("videos") if calibration_path else None
                )
                
                self.sim_env = make_env(env_config, n_envs=1)
                self.robot = SimulatedRobotWrapper(self.sim_env)
                print("ALOHA simulation initialized!")
                
            except Exception as e:
                print(f"Failed to create ALOHA simulation: {e}")
                raise
        else:
            # Real hardware
            print(f"DEBUG: robot_type = '{robot_type}'")
            
            # Default port for SO101
            default_port = "/dev/tty.usbmodem5A7A0187761"
            
            # Auto-detect or ask for port
            if robot_type in ["so101", "so101_follower"]:
                print("\nPort detection:")
                print(f"1. Use default port: {default_port}")
                print("2. Auto-detect port")
                print("3. Enter port manually")
                
                port_choice = input("Select option [1]: ").strip() or "1"
                
                if port_choice == "1":
                    port = default_port
                    print(f"Using default port: {port}")
                elif port_choice == "2":
                    print("Auto-detecting robot port...")
                    import subprocess
                    result = subprocess.run(["python", "-m", "lerobot.find_port"], capture_output=True, text=True)
                    print(result.stdout)
                    detected_ports = [line for line in result.stdout.split('\n') if '/dev/tty' in line]
                    
                    if detected_ports:
                        print("\nDetected ports:")
                        for i, p in enumerate(detected_ports):
                            print(f"{i+1}. {p}")
                        port_idx = input(f"Select port [1]: ").strip() or "1"
                        try:
                            # Extract just the port path from the output
                            port_line = detected_ports[int(port_idx)-1]
                            # Extract /dev/tty... from the line
                            import re
                            port_match = re.search(r'/dev/tty\.\w+', port_line)
                            port = port_match.group(0) if port_match else default_port
                        except:
                            port = default_port
                            print(f"Invalid selection, using default: {port}")
                    else:
                        port = default_port
                        print(f"No ports detected, using default: {port}")
                else:
                    port = input(f"Enter port [{default_port}]: ").strip() or default_port
                
                print(f"Using port: {port}")
            else:
                # For other robot types, ask for port
                port = input("Enter robot port (or press Enter to auto-detect): ").strip()
                if not port:
                    print("Auto-detecting robot port...")
                    import subprocess
                    result = subprocess.run(["python", "-m", "lerobot.find_port"], capture_output=True, text=True)
                    print(result.stdout)
                    port = input("Enter the detected port: ").strip()
            
            # Build robot config based on type
            print(f"Creating robot configuration for {robot_type}...")
            
            if robot_type == "koch" or robot_type == "koch_follower":
                robot_config = KochFollowerConfig(
                    port=port,
                    calibration_dir=Path(calibration_path or f".cache/calibration/{robot_type}")
                )
            elif robot_type == "so100" or robot_type == "so100_follower":
                robot_config = SO100FollowerConfig(
                    port=port,
                    calibration_dir=Path(calibration_path or f".cache/calibration/{robot_type}")
                )
            elif robot_type == "so101" or robot_type == "so101_follower":
                robot_config = SO101FollowerConfig(
                    port=port,
                    calibration_dir=Path(calibration_path or f".cache/calibration/{robot_type}")
                )
            else:
                raise ValueError(f"Unknown robot type: {robot_type}. Supported: koch, so100, so101")
            
            print(f"Robot configuration created: {robot_config}")
            
            # Create robot instance
            try:
                print("Creating robot instance from config...")
                self.robot = make_robot_from_config(robot_config)
                print("Robot instance created, attempting to connect...")
                
                # Check if we should skip calibration
                calibration_path = Path(calibration_path or f".cache/calibration/{robot_type}")
                skip_calibration = False
                
                if calibration_path.exists():
                    skip_cal = input("Calibration found. Skip calibration? (y/n) [y]: ").strip().lower()
                    skip_calibration = skip_cal != 'n'
                
                # Check if we should allow partial motor detection
                allow_partial = input("Allow partial motor detection? (y/n) [n]: ").strip().lower() == 'y'
                
                if allow_partial:
                    print("WARNING: Running with partial motors - some functions may not work correctly")
                    # Temporarily disable motor checking
                    import unittest.mock
                    with unittest.mock.patch.object(self.robot.bus, '_assert_motors_exist'):
                        self.robot.connect(calibrate=not skip_calibration)
                else:
                    self.robot.connect(calibrate=not skip_calibration)
                
                if skip_calibration:
                    print("Skipped calibration, using existing calibration data")
                else:
                    print("Calibration completed")
                
                print(f"Robot connected successfully!")
                print(f"Robot is_connected: {self.robot.is_connected}")
                print(f"Robot is_calibrated: {self.robot.is_calibrated}")
                print(f"Observation features: {self.robot.observation_features}")
                print(f"Action features: {self.robot.action_features}")
                
            except Exception as e:
                print(f"Failed to initialize robot hardware: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                print("\nMake sure:")
                print("1. Robot is connected and powered on")
                print("2. You have the correct permissions (may need sudo)")
                print("3. Motors are properly configured (run: python -m lerobot.setup_motors)")
                raise
        
        # Initialize teleoperator if specified
        self.teleop = None
        if teleop_type and TELEOP_AVAILABLE:
            try:
                print(f"\nInitializing {teleop_type} teleoperator...")
                
                # Get teleop port
                teleop_port = input("Enter teleoperator port (or press Enter to skip): ").strip()
                if not teleop_port:
                    print("Skipping teleoperator initialization")
                    self.teleop = None
                else:
                    # Create teleop config based on type
                    if teleop_type == "so100_leader":
                        teleop_config = SO100LeaderConfig(
                            port=teleop_port,
                            calibration_dir=Path(calibration_path or f".cache/calibration/{teleop_type}")
                        )
                    elif teleop_type == "so101_leader":
                        teleop_config = SO101LeaderConfig(
                            port=teleop_port,
                            calibration_dir=Path(calibration_path or f".cache/calibration/{teleop_type}")
                        )
                    else:
                        print(f"Unknown teleop type: {teleop_type}")
                        self.teleop = None
                        return
                    
                    self.teleop = make_teleoperator_from_config(teleop_config)
                    self.teleop.connect()
                    print("Teleoperator connected!")
            except Exception as e:
                print(f"Warning: Could not initialize teleoperator: {e}")
                self.teleop = None
        
        # Initialize cameras
        print("\nInitializing cameras...")
        self.cameras = {}
        self.has_depth = False
        self.cap = None
        self.cap_depth = None
        
        # Try to find and initialize camera with depth support
        camera_name = "front"
        camera_found = False
        
        # Try each camera index
        for idx in [0, 1, 2]:  # Changed order - try camera 0 first
            try:
                print(f"Trying camera {idx}...")
                
                # Try to open as RGB-D camera with OpenNI2 backend
                cap = cv2.VideoCapture(idx, cv2.CAP_OPENNI2)
                
                if not cap.isOpened():
                    # Try regular OpenCV
                    cap = cv2.VideoCapture(idx)
                
                if cap.isOpened():
                    # Check if depth is available
                    if cap.get(cv2.CAP_PROP_OPENNI2_SYNC) != -1:
                        # This is an RGB-D camera
                        print(f"✓ RGB-D camera detected on index {idx}!")
                        self.has_depth = True
                        
                        # Test read
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Test depth
                            if cap.grab():
                                ret_depth, depth_frame = cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
                                if ret_depth and depth_frame is not None:
                                    print(f"✓ Depth available! Shape: {depth_frame.shape}")
                                    self.cap = cap
                                    camera_found = True
                                    break
                    else:
                        # Regular camera
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"✓ Camera {idx} works (no depth)! Shape: {frame.shape}")
                            self.cap = cap
                            camera_found = True
                            self.has_depth = False
                            break
                        else:
                            cap.release()
            except Exception as e:
                print(f"Camera {idx} error: {e}")
                if cap and cap.isOpened():
                    cap.release()
        
        # If no depth yet, check for separate depth stream
        if camera_found and not self.has_depth:
            print("\nChecking for separate depth stream...")
            for idx in range(3):
                try:
                    cap_test = cv2.VideoCapture(idx + 1)
                    if cap_test.isOpened():
                        ret, test_frame = cap_test.read()
                        if ret and test_frame is not None and len(test_frame.shape) == 2:
                            print(f"✓ Found potential depth stream at index {idx + 1}")
                            self.cap_depth = cap_test
                            self.has_depth = True
                            break
                        else:
                            cap_test.release()
                except:
                    pass
        
        # Create camera wrapper
        if camera_found:
            class CVWrapper:
                def __init__(self, cap, cap_depth=None, has_depth=False):
                    self.cap = cap
                    self.cap_depth = cap_depth
                    self.has_depth = has_depth
                    
                def read(self):
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        return frame if ret else None
                    return None
                
                def read_depth(self):
                    if not self.has_depth:
                        return None
                        
                    if self.cap_depth:
                        # Separate depth stream
                        ret, depth = self.cap_depth.read()
                        return depth if ret else None
                    else:
                        # Same stream
                        if self.cap.grab():
                            ret, depth = self.cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
                            return depth if ret else None
                    return None
                    
                def __del__(self):
                    if hasattr(self, 'cap') and self.cap:
                        self.cap.release()
                    if hasattr(self, 'cap_depth') and self.cap_depth:
                        self.cap_depth.release()
            
            self.cameras = {camera_name: CVWrapper(self.cap, self.cap_depth, self.has_depth)}
            print(f"Camera initialized with depth support: {self.has_depth}")
        else:
            print("No camera available. Running without camera.")
            # Create a dummy window for keyboard input
            cv2.namedWindow('Robot Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Robot Control', 400, 300)
        
        # Load SmolVLM
        print("\nLoading vision model...")
        try:
            # Use the correct available SmolVLM model
            model_name = "HuggingFaceTB/SmolVLM-Instruct"  # The currently available model
            
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vlm = self.vlm.to(device)
            self.device = device
            self.vlm.eval()
            
            print(f"SmolVLM loaded successfully on {device}!")
            self.use_blip = False
        except Exception as e:
            print(f"Could not load SmolVLM: {e}")
            print("Loading BLIP as vision model...")
            try:
                # Use BLIP which is downloading successfully
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                model_name = "Salesforce/blip-image-captioning-base"
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.vlm = BlipForConditionalGeneration.from_pretrained(model_name)
                self.device = "cpu"
                self.vlm.eval()
                print("BLIP vision model loaded successfully!")
                self.use_blip = True
            except Exception as e2:
                print(f"Error loading vision models: {e2}")
                print("Continuing without vision model...")
                self.processor = None
                self.vlm = None
                self.use_blip = False
                self.device = "cpu"
        
        # Dataset for recording
        self.dataset = None
        self.current_episode = []
        
        # Load camera calibration if available
        self.camera_calibration = None
        self.load_camera_calibration()
        
    def auto_calibrate_camera(self):
        """Automatic camera calibration using robot arm positions"""
        if not self.cameras:
            print("No camera available for calibration!")
            return False
            
        print("\nAutomatic Camera Calibration")
        print("="*50)
        print("The robot will move to several positions.")
        print("The system will detect the gripper automatically.")
        print("\nMake sure:")
        print("1. The workspace is clear")
        print("2. The camera can see the robot gripper")
        print("3. Good lighting conditions")
        print("\nPress Enter to start or 'q' to cancel...")
        
        if input().strip().lower() == 'q':
            return False
        
        calibration_points = []
        cam_name = list(self.cameras.keys())[0]
        
        # Define positions to move to (in mm from base)
        # These are relative positions that should be visible in camera
        positions = [
            (-100, 250, "Left Near"),
            (100, 250, "Right Near"),
            (100, 350, "Right Far"),
            (-100, 350, "Left Far"),
            (0, 300, "Center"),
            (-150, 300, "Far Left"),
            (150, 300, "Far Right"),
            (0, 200, "Close Center"),
        ]
        
        # Open gripper for better visibility
        print("Opening gripper for visibility...")
        self.robot.send_action({'gripper.pos': 80.0})
        time.sleep(1)
        
        for i, (x, y, name) in enumerate(positions):
            print(f"\nPosition {i+1}/{len(positions)}: {name} ({x}, {y})mm")
            
            try:
                # Calculate rough joint positions for SO101
                # This is simplified - adjust based on your robot's kinematics
                distance = np.sqrt(x**2 + y**2)
                angle = np.arctan2(x, y) * 180 / np.pi
                
                action = {
                    'shoulder_pan.pos': angle,  # Pan to face the position
                    'shoulder_lift.pos': -30.0 - (distance / 10),  # Adjust based on distance
                    'elbow_flex.pos': -45.0,
                    'wrist_flex.pos': -45.0,
                    'wrist_roll.pos': 0.0,
                    'gripper.pos': 80.0  # Keep gripper open
                }
                
                # Send action
                self.robot.send_action(action)
                print("Moving... (waiting 3 seconds)")
                time.sleep(3)  # Wait for movement to complete
                
                # Capture image
                frame = self.cameras[cam_name].read()
                if frame is None:
                    print("Failed to capture image!")
                    continue
                
                # Try to detect gripper
                gripper_pixel = self.detect_gripper_in_image(frame.copy())
                
                if gripper_pixel is not None:
                    px, py = gripper_pixel
                    calibration_points.append((px, py, x, y))
                    
                    # Show detection
                    display_frame = frame.copy()
                    cv2.circle(display_frame, (px, py), 10, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"Detected at ({x}, {y})mm", (px-50, py-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Manual fallback
                    print("Automatic detection failed. Please click on the gripper.")
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Click on gripper position", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Get manual click
                    clicked_point = [None]
                    
                    def mouse_callback(event, mx, my, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            clicked_point[0] = (mx, my)
                    
                    cv2.namedWindow('Calibration')
                    cv2.setMouseCallback('Calibration', mouse_callback)
                    
                    while clicked_point[0] is None:
                        cv2.imshow('Calibration', display_frame)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC
                            cv2.destroyAllWindows()
                            return False
                    
                    px, py = clicked_point[0]
                    calibration_points.append((px, py, x, y))
                    print(f"Manual click at pixel ({px}, {py})")
                
                # Show progress
                cv2.putText(display_frame, f"Point {i+1}/{len(positions)} collected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Calibration', display_frame)
                cv2.waitKey(1000)
                
            except Exception as e:
                print(f"Error at position {name}: {e}")
                continue
        
        cv2.destroyAllWindows()
        
        # Compute calibration if we have enough points
        if len(calibration_points) >= 4:
            print(f"\nCollected {len(calibration_points)} calibration points")
            success = self._compute_homography_from_points(calibration_points)
            
            if success:
                print("✓ Calibration successful!")
                self._save_camera_calibration()
                
                # Return to home position
                print("Returning to home position...")
                self.robot.send_action({
                    'shoulder_pan.pos': 0.0,
                    'shoulder_lift.pos': -30.0,
                    'elbow_flex.pos': 0.0,
                    'wrist_flex.pos': -45.0,
                    'wrist_roll.pos': 0.0,
                    'gripper.pos': 50.0
                })
                return True
        else:
            print("Not enough calibration points collected!")
            return False
    
    def detect_gripper_in_image(self, image, depth_image=None):
        """Detect gripper in image using color/shape detection or depth"""
        try:
            if depth_image is not None and self.has_depth:
                # Use depth for better detection
                # Find the closest object (likely the gripper or object to grasp)
                # Mask out very close and very far values
                valid_depth = np.where((depth_image > 200) & (depth_image < 2000), depth_image, np.inf)
                
                # Find the closest point
                if np.any(np.isfinite(valid_depth)):
                    min_idx = np.unravel_index(np.argmin(valid_depth), valid_depth.shape)
                    
                    # Get a region around the closest point
                    y, x = min_idx
                    region_size = 50
                    y_min = max(0, y - region_size)
                    y_max = min(image.shape[0], y + region_size)
                    x_min = max(0, x - region_size)
                    x_max = min(image.shape[1], x + region_size)
                    
                    # Find center of mass in this region
                    region_depth = valid_depth[y_min:y_max, x_min:x_max]
                    if np.any(np.isfinite(region_depth)):
                        # Calculate center of mass weighted by inverse depth
                        y_coords, x_coords = np.mgrid[0:region_depth.shape[0], 0:region_depth.shape[1]]
                        weights = np.where(np.isfinite(region_depth), 1.0 / (region_depth + 1), 0)
                        
                        total_weight = weights.sum()
                        if total_weight > 0:
                            cy = int((y_coords * weights).sum() / total_weight) + y_min
                            cx = int((x_coords * weights).sum() / total_weight) + x_min
                            
                            # Draw detection on image
                            cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)
                            cv2.putText(image, f"Depth: {int(valid_depth[cy, cx])}mm", (cx-50, cy-20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            return (cx, cy)
            
            # Fallback to color detection
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Look for gripper colors (adjust these for your gripper)
            # This example looks for dark/metallic colors
            lower_bound = np.array([0, 0, 20])
            upper_bound = np.array([180, 100, 100])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (likely the gripper)
                largest = max(contours, key=cv2.contourArea)
                
                # Get center of contour
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
            
            return None
            
        except Exception as e:
            print(f"Gripper detection error: {e}")
            return None
    
    def _compute_homography_from_points(self, calibration_points):
        """Compute homography from calibration points"""
        if len(calibration_points) < 4:
            return False
        
        # Separate pixel and robot coordinates
        src_points = np.array([(p[0], p[1]) for p in calibration_points], dtype=np.float32)
        dst_points = np.array([(p[2], p[3]) for p in calibration_points], dtype=np.float32)
        
        # Compute homography
        self.camera_calibration = {}
        self.camera_calibration["homography"], mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if self.camera_calibration["homography"] is not None:
            # Test accuracy
            total_error = 0
            for px, py, rx, ry in calibration_points:
                # Transform pixel to robot
                point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(point, self.camera_calibration["homography"])
                pred_x, pred_y = transformed[0][0]
                
                error = np.sqrt((pred_x - rx)**2 + (pred_y - ry)**2)
                total_error += error
                print(f"Point ({px},{py}) -> ({rx},{ry}): error = {error:.1f}mm")
            
            avg_error = total_error / len(calibration_points)
            print(f"\nAverage calibration error: {avg_error:.1f}mm")
            
            self.camera_calibration["points"] = calibration_points
            self.camera_calibration["avg_error"] = avg_error
            
            return True
        
        return False
    
    def _save_camera_calibration(self):
        """Save calibration to file"""
        if self.camera_calibration is None:
            return
            
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "robot_type": self.robot_type,
            "homography_matrix": self.camera_calibration["homography"].tolist(),
            "calibration_points": self.camera_calibration["points"],
            "average_error": self.camera_calibration.get("avg_error", 0),
            "camera_index": 0,  # Update if using different camera
            "resolution": (640, 480)  # Update based on your camera
        }
        
        with open("camera_calibration.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print("Calibration saved to camera_calibration.json")
    
    def load_camera_calibration(self, calibration_file="camera_calibration.json"):
        """Load camera calibration if available"""
        try:
            import json
            from pathlib import Path
            
            if Path(calibration_file).exists():
                with open(calibration_file, "r") as f:
                    calib_data = json.load(f)
                
                self.camera_calibration = {
                    "homography": np.array(calib_data["homography_matrix"]),
                    "camera_index": calib_data.get("camera_index", 0),
                    "resolution": calib_data.get("resolution", (640, 480))
                }
                print(f"Loaded camera calibration from {calibration_file}")
                return True
        except Exception as e:
            print(f"Could not load calibration: {e}")
        
        self.camera_calibration = None
        return False
    
    def pixel_to_robot(self, pixel_x, pixel_y):
        """Convert pixel coordinates to robot coordinates using calibration"""
        if self.camera_calibration is None:
            return None, None
        
        point = np.array([[pixel_x, pixel_y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.camera_calibration["homography"])
        robot_x, robot_y = transformed[0][0]
        
        return robot_x, robot_y
    
    def robot_to_pixel(self, robot_x, robot_y):
        """Convert robot coordinates to pixel coordinates using calibration"""
        if self.camera_calibration is None:
            return None
            
        # Get inverse homography
        inv_h = np.linalg.inv(self.camera_calibration["homography"])
        
        point = np.array([[robot_x, robot_y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, inv_h)
        pixel_x, pixel_y = transformed[0][0]
        
        return (int(pixel_x), int(pixel_y))
    
    def calibrate_robot(self):
        """Run robot calibration procedure"""
        if self.use_sim:
            print("Simulation doesn't need calibration")
            return
            
        print("\nStarting robot calibration...")
        print("This will move the robot to find joint limits")
        print("Make sure the area is clear!")
        
        input("Press Enter to start calibration...")
        
        try:
            # Calibrate the robot
            self.robot.calibrate()
            print("Calibration complete!")
            
            # The calibration is automatically saved by the robot
            print(f"Calibration saved")
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            raise
    
    def detect_objects(self, image, depth_frame=None):
        """Use vision model to detect objects and plan actions, with depth support"""
        if self.vlm is None:
            # No vision model, but we can still use depth
            if depth_frame is not None and self.has_depth:
                controller = PickPlaceController(self)
                return controller.detect_objects_with_depth_only(image, depth_frame)
            return {'objects': [], 'targets': [], 'actions': [], 'robot_coords': []}
            
        # Your existing vision model code...
        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image
        
        try:
            if hasattr(self, 'use_blip') and self.use_blip:
                # Use BLIP for basic object detection
                inputs = self.processor(pil_img, return_tensors="pt")
                out = self.vlm.generate(**inputs, max_length=50)
                description = self.processor.decode(out[0], skip_special_tokens=True)
                print(f"BLIP vision: {description}")
                
                # Simple keyword detection
                detections = {
                    'objects': [],
                    'targets': [],
                    'actions': [],
                    'robot_coords': []
                }
                
                # Look for keywords
                if any(word in description.lower() for word in ['block', 'cube', 'object', 'toy']):
                    # Assume object in center for now
                    detections['objects'].append((320, 240))
                    if self.camera_calibration is not None:
                        robot_x, robot_y = self.pixel_to_robot(320, 240)
                        detections['robot_coords'].append((robot_x, robot_y))
                
                if 'grasp' in description.lower() or 'pick' in description.lower():
                    detections['actions'].append('grasp')
                elif 'release' in description.lower() or 'drop' in description.lower():
                    detections['actions'].append('release')
                
                return detections
            else:
                # Original SmolVLM code
                # Task-specific prompts
                if self.robot_type in ["so100", "so101"]:
                    prompt = """Analyze this robot workspace. I have an SO-101 robot arm.
                    Identify:
                    1. Any graspable objects (blocks, cups, toys)
                    2. Target locations (plates, containers, goals)
                    3. The robot gripper position
                    4. Suggested action: 'move_to: x,y', 'grasp', 'release', or 'home'
                    Return coordinates and action."""
                elif self.robot_type == "koch":
                    prompt = """Analyze this robot workspace. I have a Koch robot arm.
                    Identify:
                    1. Any graspable objects (blocks, cups, toys)
                    2. Target locations (plates, containers, goals)
                    3. The robot gripper position
                    4. Suggested action: 'move_to: x,y', 'grasp', 'release', or 'home'
                    Return coordinates and action."""
                else:
                    prompt = """Analyze this robot workspace. 
                    Identify objects, targets, and robot position.
                    Suggest next action with coordinates."""
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt_text, images=[pil_img], return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.vlm.generate(**inputs, do_sample=False, max_new_tokens=150)
                
                response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract just the assistant's response
                if "<|assistant|>" in response:
                    response = response.split("<|assistant|>")[-1].strip()
                
                print(f"Vision: {response}")
                
                # Parse response
                detections = {
                    'objects': [],
                    'targets': [],
                    'actions': [],
                    'robot_coords': []  # Add robot coordinates
                }
                
                # Extract positions
                object_matches = re.findall(r'object:\s*(\d+),\s*(\d+)', response)
                for x, y in object_matches:
                    pixel_x, pixel_y = int(x), int(y)
                    detections['objects'].append((pixel_x, pixel_y))
                    
                    # Convert to robot coordinates if calibrated
                    if self.camera_calibration is not None:
                        robot_x, robot_y = self.pixel_to_robot(pixel_x, pixel_y)
                        detections['robot_coords'].append((robot_x, robot_y))
                        print(f"Object at pixel ({pixel_x}, {pixel_y}) -> robot ({robot_x:.1f}, {robot_y:.1f})mm")
                
                target_matches = re.findall(r'target:\s*(\d+),\s*(\d+)', response)
                for x, y in target_matches:
                    detections['targets'].append((int(x), int(y)))
                
                # Extract actions
                if 'grasp' in response.lower():
                    detections['actions'].append('grasp')
                elif 'release' in response.lower():
                    detections['actions'].append('release')
                elif 'move_to' in response.lower():
                    move_matches = re.findall(r'move_to:\s*(\d+),\s*(\d+)', response)
                    if move_matches:
                        detections['actions'].append(('move_to', int(move_matches[0][0]), int(move_matches[0][1])))
                
                return detections
                
        except Exception as e:
            print(f"Vision error: {e}")
            return {'objects': [], 'targets': [], 'actions': [], 'robot_coords': []}
    
    def vision_to_joint_action(self, detections, current_obs):
        """Convert vision detections to joint actions"""
        # Build action dict matching robot's action features
        action = {}
        
        # Get action names from robot
        action_names = list(self.robot.action_features.keys())
        
        # Initialize all actions to zero
        for name in action_names:
            action[name] = 0.0
        
        if detections['actions']:
            action_info = detections['actions'][0]
            
            # Simple action mapping - customize based on your robot
            if action_info == 'grasp':
                # Close gripper - usually last action
                if 'gripper' in action_names:
                    action['gripper'] = -1.0
                elif len(action_names) > 0:
                    action[action_names[-1]] = -1.0
                    
            elif action_info == 'release':
                # Open gripper
                if 'gripper' in action_names:
                    action['gripper'] = 1.0
                elif len(action_names) > 0:
                    action[action_names[-1]] = 1.0
                    
            elif isinstance(action_info, tuple) and action_info[0] == 'move_to':
                # Convert pixel coordinates to joint movements
                _, target_x, target_y = action_info
                
                # Simple proportional control (needs proper kinematics)
                if len(action_names) >= 3:
                    action[action_names[0]] = (target_x - 320) / 320 * 0.1
                    action[action_names[1]] = (240 - target_y) / 240 * 0.1
                    action[action_names[2]] = -0.05
        
        return action
    
    def teleop_mode(self):
        """Teleoperation mode - control follower with leader arms"""
        if self.teleop is None:
            print("No teleoperator configured!")
            return []
            
        print("\nTeleoperation Mode")
        print("Move the leader arm to control the follower arm")
        print("Press 'q' to quit, 'r' to record episode")
        
        recording = False
        episode_data = []
        fps = 30
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                # Get observation from robot
                observation = self.robot.get_observation()
                
                # Get action from teleoperator
                action = self.teleop.get_action()
                
                # Send action to robot
                sent_action = self.robot.send_action(action)
                
                # Show camera feed
                for cam_name, camera in self.cameras.items():
                    frame = camera.read()
                    if frame is not None and cam_name == "top" or cam_name == "front":
                        cv2.imshow(f'Camera: {cam_name}', frame)
                
                # Record if enabled
                if recording:
                    # Combine observation with camera images
                    full_obs = observation.copy()
                    for cam_name, camera in self.cameras.items():
                        frame = camera.read()
                        if frame is not None:
                            full_obs[f"observation.images.{cam_name}"] = frame
                    
                    episode_data.append({
                        'observation': full_obs,
                        'action': sent_action,
                        'timestamp': time.time()
                    })
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    recording = not recording
                    if recording:
                        print("Recording started...")
                        episode_data = []
                    else:
                        print(f"Recording stopped. {len(episode_data)} frames captured.")
                
                # Maintain FPS
                dt_s = time.perf_counter() - loop_start
                if dt_s < 1/fps:
                    time.sleep(1/fps - dt_s)
                    
        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
        finally:
            cv2.destroyAllWindows()
            
        return episode_data
    
    def autonomous_mode(self):
        """Enhanced autonomous mode with pick and place capability"""
        print("\nEnhanced Autonomous Pick & Place Mode")
        print("="*50)
        print("The robot will automatically:")
        print("1. Search for objects using depth/vision")
        print("2. Pick up the nearest object")
        print("3. Move it to a designated drop zone")
        print("\nPress 'q' to quit, 'space' to pause, 'r' to reset")
        print("Press 'c' to auto-calibrate camera")
        
        controller = PickPlaceController(self)
        paused = False
        fps = 10  # Slower for pick and place
        
        # Check if camera calibration exists
        if self.camera_calibration is None:
            print("\nNo camera calibration found!")
            cal = input("Run auto-calibration now? (y/n) [y]: ").strip().lower()
            if cal != 'n':
                success = self.auto_calibrate_camera()
                if not success:
                    print("Calibration failed. Running without calibration...")
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                if not paused:
                    # Get observation
                    observation = self.robot.get_observation()
                    
                    # Get camera image and depth
                    cam_name = list(self.cameras.keys())[0] if self.cameras else None
                    if cam_name:
                        frame = self.cameras[cam_name].read()
                        depth_frame = None
                        
                        if self.has_depth and hasattr(self.cameras[cam_name], 'read_depth'):
                            depth_frame = self.cameras[cam_name].read_depth()
                        
                        if frame is not None:
                            display_frame = frame.copy()
                            
                            # Get detections
                            if self.vlm is not None:
                                detections = self.detect_objects(display_frame, depth_frame)
                            else:
                                # Use depth-only detection
                                detections = controller.detect_objects_with_depth_only(display_frame, depth_frame)
                            
                            # Draw current state
                            cv2.putText(display_frame, f"State: {controller.state}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            if controller.target_object:
                                cv2.putText(display_frame, f"Target: ({controller.target_object[0]:.0f}, {controller.target_object[1]:.0f})mm", 
                                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Draw drop zone
                            if self.camera_calibration is not None:
                                # Convert drop location to pixel coordinates
                                drop_pixel = self.robot_to_pixel(controller.drop_location[0], controller.drop_location[1])
                                if drop_pixel:
                                    cv2.circle(display_frame, drop_pixel, 20, (255, 0, 255), 2)
                                    cv2.putText(display_frame, "DROP", (drop_pixel[0]-20, drop_pixel[1]-25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            cv2.imshow('Pick & Place Vision', display_frame)
                            
                            # Execute pick and place
                            action = controller.execute_pick_place(detections)
                            
                            if action:
                                # Apply safety limits
                                for key in action:
                                    if key != 'gripper.pos':
                                        action[key] = np.clip(action[key], -90, 90)
                                
                                # Send action
                                self.robot.send_action(action)
                
                # Handle keyboard
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('r'):
                    controller.state = "SEARCHING"
                    controller.target_object = None
                    print("Reset to searching state")
                elif key == ord('c'):
                    self.auto_calibrate_camera()
                
                # Maintain FPS
                dt_s = time.perf_counter() - loop_start
                if dt_s < 1/fps:
                    time.sleep(1/fps - dt_s)
                    
        except KeyboardInterrupt:
            print("\nPick & place stopped.")
        finally:
            cv2.destroyAllWindows()
    
    def collect_dataset(self, task_name="pick_place", num_episodes=10):
        """Collect a dataset for training"""
        print(f"\nCollecting dataset: {task_name}")
        print(f"Target: {num_episodes} episodes")
        
        if self.teleop is None:
            print("Error: No teleoperator configured for data collection!")
            return None
        
        # Create dataset
        repo_id = f"lerobot_{self.robot_type}_{task_name}"
        root_path = Path(f"data/{repo_id}")
        root_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset with proper features
        features = {
            **self.robot.observation_features,
            **self.robot.action_features
        }
        
        # Add camera features
        for cam_name in self.cameras:
            features[f"observation.images.{cam_name}"] = {"shape": (480, 640, 3), "dtype": "uint8"}
        
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=30,
            root=root_path,
            features=features,
            force_override=True
        )
        
        # Collect episodes
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("Demonstrate the task using teleoperation...")
            
            # Collect episode via teleoperation
            episode_data = self.teleop_mode()
            
            if episode_data:
                # Add episode to dataset
                for frame in episode_data:
                    self.dataset.add_frame({
                        **frame['observation'],
                        **frame['action'],
                        'timestamp': frame['timestamp']
                    })
                
                print(f"Episode {episode + 1} saved ({len(episode_data)} frames)")
        
        # Save dataset
        self.dataset.save_to_disk()
        print(f"\nDataset saved to {root_path}")
        
        return self.dataset
    
    def pose_arm(self):
        """Move arm to predefined poses"""
        print("\nArm Posing Mode")
        print("="*50)
        print("Poses:")
        print("  'h' - Home position")
        print("  'r' - Ready position")
        print("  'p' - Pick position")
        print("  'd' - Drop position")
        print("  'w' - Wave")
        print("  'q' - Quit posing mode")
        print("="*50)
        
        # Define poses (adjust these values for your SO101)
        poses = {
            'h': {  # Home
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -30.0,
                'elbow_flex.pos': 0.0,
                'wrist_flex.pos': -45.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': 50.0
            },
            'r': {  # Ready
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -45.0,
                'elbow_flex.pos': -45.0,
                'wrist_flex.pos': -45.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': 80.0  # Open gripper
            },
            'p': {  # Pick
                'shoulder_pan.pos': 0.0,
                'shoulder_lift.pos': -60.0,
                'elbow_flex.pos': -90.0,
                'wrist_flex.pos': -30.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': 80.0  # Open gripper
            },
            'd': {  # Drop
                'shoulder_pan.pos': 45.0,
                'shoulder_lift.pos': -45.0,
                'elbow_flex.pos': -45.0,
                'wrist_flex.pos': -45.0,
                'wrist_roll.pos': 0.0,
                'gripper.pos': 20.0  # Closed gripper
            }
        }
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("Moving to home position...")
                self.robot.send_action(poses['h'])
            elif key == ord('r'):
                print("Moving to ready position...")
                self.robot.send_action(poses['r'])
            elif key == ord('p'):
                print("Moving to pick position...")
                self.robot.send_action(poses['p'])
            elif key == ord('d'):
                print("Moving to drop position...")
                self.robot.send_action(poses['d'])
            elif key == ord('w'):
                # Wave animation
                print("Waving...")
                for i in range(3):
                    wave_left = poses['h'].copy()
                    wave_left['shoulder_pan.pos'] = -30.0
                    self.robot.send_action(wave_left)
                    time.sleep(0.5)
                    
                    wave_right = poses['h'].copy()
                    wave_right['shoulder_pan.pos'] = 30.0
                    self.robot.send_action(wave_right)
                    time.sleep(0.5)
                
                self.robot.send_action(poses['h'])
            
            # Show current position
            obs = self.robot.get_observation()
            print(f"Current position: {obs}")
    
    def run_interactive(self):
        """Run robot with interactive control"""
        print("\nInteractive Robot Control")
        print("="*50)
        print("Controls:")
        if self.teleop:
            print("  't' - Teleoperation mode")
        print("  'a' - Autonomous pick & place mode")
        print("  'p' - Pose arm (predefined positions)")
        print("  'v' - Test vision system")
        print("  'o' - Show observation")
        print("  'g' - Toggle gripper")
        print("  'c' - Auto-calibrate camera")
        print("  'q' - Quit")
        print("="*50)
        
        fps = 30
        gripper_open = True
        
        while True:
            loop_start = time.perf_counter()
            
            # Show camera feed if available
            if self.cameras:
                cam_name = list(self.cameras.keys())[0]
                frame = self.cameras[cam_name].read()
                if frame is not None:
                    # Add text overlay
                    cv2.putText(frame, "SO101 Robot Control", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'h' for help", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow('Robot Camera', frame)
            else:
                # No camera - create a blank window for keyboard input
                blank = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(blank, "SO101 Robot Control", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(blank, "No Camera Found", (80, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(blank, "Press 'p' for poses", (80, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank, "Press 'h' for help", (80, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('Robot Control', blank)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t') and self.teleop:
                self.teleop_mode()
            elif key == ord('a'):
                self.autonomous_mode()
            elif key == ord('p'):
                self.pose_arm()
            elif key == ord('v'):
                # Test vision
                if self.cameras and self.vlm is not None:
                    cam_name = list(self.cameras.keys())[0]
                    frame = self.cameras[cam_name].read()
                    if frame is not None:
                        detections = self.detect_objects(frame)
                        print(f"Vision detections: {detections}")
                else:
                    print("No camera or vision model available")
            elif key == ord('o'):
                # Show observation
                obs = self.robot.get_observation()
                print(f"Current observation: {obs}")
            elif key == ord('g'):
                # Toggle gripper
                gripper_open = not gripper_open
                gripper_pos = 80.0 if gripper_open else 20.0
                print(f"Gripper {'opening' if gripper_open else 'closing'}...")
                self.robot.send_action({'gripper.pos': gripper_pos})
            elif key == ord('c'):
                # Run camera calibration
                self.auto_calibrate_camera()
            elif key == ord('h'):
                # Show help
                print("\nControls:")
                print("  q - Quit")
                print("  a - Autonomous pick & place")
                print("  p - Pose arm")
                print("  v - Test vision")
                print("  o - Show observation")
                print("  g - Toggle gripper")
                print("  c - Calibrate camera")
                if self.teleop:
                    print("  t - Teleoperation")
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_start
            if dt_s < 1/fps:
                time.sleep(1/fps - dt_s)
        
        cv2.destroyAllWindows()

def main():
    print("\nLeRobot Hardware Interface")
    print("="*50)
    print("Available robots:", ["koch", "so100", "so101", "aloha-sim"])
    
    # Get robot type
    robot_type_input = input("Enter robot type [so101]: ").strip() or "so101"
    
    # Check if it's simulation
    use_sim = False
    if robot_type_input == "aloha-sim":
        robot_type = "aloha"
        use_sim = True
    else:
        robot_type = robot_type_input
    
    # Check if teleoperation is desired (not for simulation)
    teleop_type = None
    if not use_sim:
        use_teleop = input("Enable teleoperation? (y/n) [n]: ").strip().lower() == 'y'
        
        if use_teleop and TELEOP_AVAILABLE:
            # Map robot type to teleop type
            teleop_map = {
                "koch": "koch_leader",
                "so100": "so100_leader", 
                "so101": "so101_leader"
            }
            teleop_type = teleop_map.get(robot_type)
            if not teleop_type or teleop_type == "koch_leader":
                print(f"Note: Teleoperation for {robot_type} may require custom setup")
                teleop_type = None
    
    try:
        robot = LeRobotHardwareWithVision(
            robot_type=robot_type,
            teleop_type=teleop_type,
            use_sim=use_sim
        )
    except Exception as e:
        print(f"\nFailed to initialize robot: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("LeRobot Hardware Control")
        print("="*50)
        print("1. Calibrate robot")
        print("2. Interactive control")
        print("3. Collect dataset")
        print("4. Test robot connection")
        print("5. Find cameras")
        print("6. Camera calibration")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            robot.calibrate_robot()
        elif choice == "2":
            robot.run_interactive()
        elif choice == "3":
            task = input("Task name [pick_place]: ").strip() or "pick_place"
            episodes = int(input("Number of episodes [5]: ").strip() or "5")
            robot.collect_dataset(task_name=task, num_episodes=episodes)
        elif choice == "4":
            # Test robot
            print("\nTesting robot connection...")
            print(f"Connected: {robot.robot.is_connected}")
            print(f"Calibrated: {robot.robot.is_calibrated}")
            obs = robot.robot.get_observation()
            print(f"Sample observation: {obs}")
        elif choice == "5":
            # List available cameras
            import subprocess
            print("\nFinding cameras...")
            subprocess.run(["python", "-m", "lerobot.find_cameras"])
        elif choice == "6":
            # Run automatic camera calibration
            print("\nCamera Calibration Options:")
            print("1. Automatic (using robot arm)")
            print("2. Manual (using markers)")
            
            cal_choice = input("Select method [1]: ").strip() or "1"
            
            if cal_choice == "1":
                robot.auto_calibrate_camera()
            else:
                # Run manual calibration
                cam_idx = input("Enter camera index to calibrate [0]: ").strip() or "0"
                mode = input("Calibration mode (quick/advanced/test) [quick]: ").strip() or "quick"
                
                import subprocess
                subprocess.run(["python", "camera_calibration.py", "--camera", cam_idx, "--mode", mode])
            
            # Reload calibration
            robot.load_camera_calibration()
        elif choice == "0":
            break
        else:
            print("Invalid option")
    
    # Cleanup
    robot.robot.disconnect()
    if robot.teleop:
        robot.teleop.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
(base) jakehennessy@MacBookAir robot-hack % 
