#!/usr/bin/env python3
"""
Real LeRobot Hardware Integration with SmolVLM for vision-based control
Works with actual LeRobot hardware (Koch, ALOHA, SO100, SO101, etc.)

Requirements:
- pip install lerobot
- pip install transformers
- pip install opencv-python pillow
- Hardware: LeRobot-compatible robot (Koch, ALOHA, SO100, etc.)
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
            
            # Use hardcoded port for SO101, ask for others
            if robot_type in ["so101", "so101_follower"]:
                port = "/dev/tty.usbmodem5A7A0186301"
                print(f"Using hardcoded port for SO101: {port}")
            else:
                # First, find the robot port
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
                
                self.robot.connect()
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
        try:
            # Get camera indices based on robot type
            camera_indices = self.get_camera_indices()
            
            # Create camera configs
            camera_configs = {}
            for cam_name, cam_idx in camera_indices.items():
                camera_configs[cam_name] = OpenCVCameraConfig(
                    camera_index=cam_idx,
                    fps=30,
                    width=640,
                    height=480,
                    color_mode="rgb"
                )
            
            # Create cameras from configs
            self.cameras = make_cameras_from_configs(camera_configs)
            
            # Start cameras
            for camera in self.cameras.values():
                camera.start()
                
            print(f"Cameras initialized: {list(self.cameras.keys())}")
        except Exception as e:
            print(f"Warning: Could not initialize cameras: {e}")
            self.cameras = {}
        
        # Load SmolVLM
        print("\nLoading SmolVLM2...")
        try:
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")
            self.vlm.eval()
            print("SmolVLM loaded!")
        except Exception as e:
            print(f"Error loading SmolVLM: {e}")
            print("Continuing without vision model...")
            self.processor = None
            self.vlm = None
        
        # Dataset for recording
        self.dataset = None
        self.current_episode = []
        
        # Load camera calibration if available
        self.load_camera_calibration()
        
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
                    "camera_index": calib_data["camera_index"],
                    "resolution": calib_data["resolution"]
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
        """Get camera indices based on robot type"""
        # Allow custom camera configuration
        print("\nCamera Configuration:")
        print("Enter camera indices (press Enter to use defaults)")
        
        cameras = {}
        
        if self.robot_type == "koch":
            # Ask for camera index
            default_cam = input("Enter main camera index [2]: ").strip()
            cam_idx = int(default_cam) if default_cam else 2
            cameras["top"] = cam_idx
        elif self.robot_type == "aloha":
            top_cam = input("Enter top camera index [0]: ").strip()
            cameras["top"] = int(top_cam) if top_cam else 0
            
            left_cam = input("Enter left wrist camera index [1]: ").strip()
            if left_cam:
                cameras["left_wrist"] = int(left_cam)
                
            right_cam = input("Enter right wrist camera index [2]: ").strip()
            if right_cam:
                cameras["right_wrist"] = int(right_cam)
        else:
            # Default single camera
            default_cam = input("Enter camera index [0]: ").strip()
            cam_idx = int(default_cam) if default_cam else 0
            cameras["front"] = cam_idx
            
        return cameras
    
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
    
    def detect_objects(self, image):
        """Use SmolVLM to detect objects and plan actions"""
        if self.vlm is None:
            return {'objects': [], 'targets': [], 'actions': []}
            
        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image
        
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
        
        try:
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[pil_img], return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.vlm.generate(**inputs, do_sample=False, max_new_tokens=150)
            
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
            return {'objects': [], 'targets': [], 'actions': []}
    
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
        """Autonomous mode with vision guidance"""
        print("\nAutonomous Mode with Vision")
        print("Robot will use vision to perform tasks")
        print("Press 'q' to quit, 'space' to pause")
        
        paused = False
        fps = 30
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                if not paused:
                    # Get observation
                    observation = self.robot.get_observation()
                    
                    # Get camera image
                    cam_name = list(self.cameras.keys())[0] if self.cameras else None
                    if cam_name:
                        frame = self.cameras[cam_name].read()
                        if frame is not None:
                            # Show frame
                            cv2.imshow('Robot Vision', frame)
                            
                            # Get vision detections
                            detections = self.detect_objects(frame)
                            
                            # Compute action
                            action = self.vision_to_joint_action(detections, observation)
                            
                            # Apply safety limits
                            for key in action:
                                action[key] = np.clip(action[key], -0.1, 0.1)
                            
                            # Send action
                            self.robot.send_action(action)
                
                # Handle keyboard
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                
                # Maintain FPS
                dt_s = time.perf_counter() - loop_start
                if dt_s < 1/fps:
                    time.sleep(1/fps - dt_s)
                    
        except KeyboardInterrupt:
            print("\nAutonomous mode stopped.")
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
    
    def run_interactive(self):
        """Run robot with interactive control"""
        print("\nInteractive Robot Control")
        print("="*50)
        print("Controls:")
        if self.teleop:
            print("  't' - Teleoperation mode")
        print("  'a' - Autonomous mode (vision-guided)")
        print("  'v' - Test vision system")
        print("  'o' - Show observation")
        print("  'q' - Quit")
        print("="*50)
        
        fps = 30
        
        while True:
            loop_start = time.perf_counter()
            
            # Show camera feed
            cam_name = list(self.cameras.keys())[0] if self.cameras else None
            if cam_name:
                frame = self.cameras[cam_name].read()
                if frame is not None:
                    cv2.imshow('Robot Camera', frame)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t') and self.teleop:
                self.teleop_mode()
            elif key == ord('a'):
                self.autonomous_mode()
            elif key == ord('v'):
                # Test vision
                if cam_name and self.vlm is not None:
                    frame = self.cameras[cam_name].read()
                    if frame is not None:
                        detections = self.detect_objects(frame)
                        print(f"Vision detections: {detections}")
            elif key == ord('o'):
                # Show observation
                obs = self.robot.get_observation()
                print(f"Current observation: {obs}")
            
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
            # Run camera calibration
            print("\nCamera Calibration")
            cam_idx = input("Enter camera index to calibrate [2]: ").strip() or "2"
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
