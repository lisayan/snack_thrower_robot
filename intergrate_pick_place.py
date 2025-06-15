#!/usr/bin/env python3
"""
LeRobot Bag Picking Integration
Uses LeRobot's framework for data collection and robot control
with SmolVLM2 for bag detection
"""

import time
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import re
from transformers import AutoProcessor, AutoModelForImageTextToText

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import add_frame, create_lerobot_dataset
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.utils import get_arm_id
from lerobot.common.cameras.camera import Camera
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config

# Configuration
REALSENSE_INDEX = 0  # Your RealSense D455
VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATASET_NAME = "bag_pick_demonstrations"
FPS = 30

class LeRobotBagPicker:
    """Bag picking system using LeRobot framework"""
    
    def __init__(self, robot_type="xarm", use_vlm=True, record_data=True):
        self.robot_type = robot_type
        self.use_vlm = use_vlm
        self.record_data = record_data
        self.episode_index = 0
        
        # Initialize robot using LeRobot's robot factory
        print("Initializing robot...")
        self.robot = self._init_robot()
        
        # Initialize camera using LeRobot's camera system
        print("Initializing camera...")
        self.camera = self._init_camera()
        
        # Initialize VLM for detection
        if self.use_vlm:
            print("Loading SmolVLM2...")
            self.processor = AutoProcessor.from_pretrained(VLM_MODEL)
            self.model = AutoModelForImageTextToText.from_pretrained(
                VLM_MODEL,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")
            self.model.eval()
        
        # Initialize dataset for recording
        if self.record_data:
            self.dataset = self._init_dataset()
        
        # State tracking
        self.current_episode = []
        self.success_count = 0
        self.total_attempts = 0
        
    def _init_robot(self):
        """Initialize robot using LeRobot's configuration"""
        if self.robot_type == "xarm":
            # Custom xArm configuration for LeRobot
            from lerobot.configs.robot.xarm import XArmConfig
            robot_cfg = XArmConfig()
            robot_cfg.ip = "192.168.1.100"  # Update with your IP
            return Robot(robot_cfg)
        else:
            # Use LeRobot's default robot initialization
            robot_cfg = init_hydra_config(f"lerobot/configs/robot/{self.robot_type}.yaml")
            return Robot(**robot_cfg)
    
    def _init_camera(self):
        """Initialize camera using LeRobot's camera system"""
        # Try to use LeRobot's camera wrapper
        try:
            from lerobot.common.cameras.opencv import OpenCVCamera
            camera = OpenCVCamera(camera_index=REALSENSE_INDEX)
            camera.fps = FPS
            return camera
        except:
            # Fallback to basic OpenCV
            class SimpleCamera:
                def __init__(self, index):
                    self.cap = cv2.VideoCapture(index)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.fps = FPS
                
                def get_frame(self):
                    ret, frame = self.cap.read()
                    return frame if ret else None
            
            return SimpleCamera(REALSENSE_INDEX)
    
    def _init_dataset(self):
        """Initialize LeRobot dataset for recording demonstrations"""
        dataset_path = Path(f"data/{DATASET_NAME}")
        
        # Create dataset with LeRobot format
        dataset = create_lerobot_dataset(
            repo_id=DATASET_NAME,
            fps=FPS,
            robot_type=self.robot_type,
            camera_names=["realsense"],
            root=dataset_path,
            force_override=False
        )
        
        return dataset
    
    def detect_bag(self, frame):
        """Detect bag using SmolVLM2"""
        if not self.use_vlm:
            # Simple color detection fallback
            return self._detect_by_color(frame)
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        prompt = "Find the bag on the table and return only its center coordinates as 'x,y'. If no bag visible, return '-1,-1'."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        try:
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[img], return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)
            
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            match = re.search(r'(\d+)\s*,\s*(\d+)', response)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return x, y
        except Exception as e:
            print(f"Detection error: {e}")
        
        return None, None
    
    def _detect_by_color(self, frame):
        """Fast color-based detection fallback"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for bags (adjust based on your bags)
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        
        return None, None
    
    def pixel_to_robot_coords(self, px, py, frame_shape):
        """Convert pixel to robot coordinates"""
        h, w = frame_shape[:2]
        
        # Simple linear mapping (calibrate for your setup!)
        # These values need calibration for your specific setup
        WORKSPACE_WIDTH = 600  # mm
        WORKSPACE_DEPTH = 400  # mm
        CAMERA_OFFSET_X = 0    # mm
        CAMERA_OFFSET_Y = 300  # mm
        
        # Normalize pixel coordinates
        norm_x = (px - w/2) / (w/2)
        norm_y = (py - h/2) / (h/2)
        
        # Convert to robot coordinates
        robot_x = norm_x * (WORKSPACE_WIDTH/2) + CAMERA_OFFSET_X
        robot_y = -norm_y * (WORKSPACE_DEPTH/2) + CAMERA_OFFSET_Y
        
        return robot_x, robot_y
    
    def execute_pick_sequence(self, target_x, target_y):
        """Execute pick sequence and record data"""
        print(f"Executing pick at ({target_x:.1f}, {target_y:.1f})mm")
        
        # Define waypoints
        waypoints = [
            # Current position
            self.robot.get_state(),
            # Pre-grasp position
            {"x": target_x, "y": target_y, "z": 200, "gripper": 1.0},
            # Grasp position
            {"x": target_x, "y": target_y, "z": 50, "gripper": 1.0},
            # Close gripper
            {"x": target_x, "y": target_y, "z": 50, "gripper": 0.0},
            # Lift
            {"x": target_x, "y": target_y, "z": 200, "gripper": 0.0},
            # Move to drop zone
            {"x": -200, "y": 200, "z": 200, "gripper": 0.0},
            # Open gripper
            {"x": -200, "y": 200, "z": 200, "gripper": 1.0},
        ]
        
        # Execute trajectory
        episode_data = []
        start_time = time.time()
        
        for i, waypoint in enumerate(waypoints):
            # Move robot
            if self.robot_type == "xarm":
                # Custom xArm control
                self.robot.arm.set_position(
                    x=waypoint["x"], 
                    y=waypoint["y"], 
                    z=waypoint["z"],
                    wait=True
                )
                self.robot.arm.set_gripper_position(
                    waypoint["gripper"] * 800,  # Convert to xArm scale
                    wait=True
                )
            else:
                # LeRobot standard control
                self.robot.move_to(waypoint)
            
            # Record data if enabled
            if self.record_data:
                frame = self.camera.get_frame()
                robot_state = self.robot.get_state()
                
                data_point = {
                    "timestamp": time.time() - start_time,
                    "image": frame,
                    "robot_state": robot_state,
                    "action": waypoint,
                    "step": i,
                }
                episode_data.append(data_point)
        
        return episode_data
    
    def record_episode(self, episode_data, success):
        """Record episode to LeRobot dataset"""
        if not self.record_data or not episode_data:
            return
        
        print(f"Recording episode {self.episode_index} (success: {success})")
        
        # Add episode to dataset
        for i, data in enumerate(episode_data):
            add_frame(
                dataset=self.dataset,
                episode_index=self.episode_index,
                frame_index=i,
                image=data["image"],
                state=data["robot_state"],
                action=data["action"],
                success=success
            )
        
        # Save dataset
        self.dataset.save_episode(self.episode_index)
        self.episode_index += 1
    
    def autonomous_mode(self, num_episodes=10):
        """Run autonomous bag picking with data collection"""
        print(f"\nStarting autonomous mode for {num_episodes} episodes")
        print("=" * 50)
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Reset robot to home
            print("Moving to home position...")
            if self.robot_type == "xarm":
                self.robot.arm.set_position(x=0, y=300, z=300, wait=True)
            else:
                self.robot.reset()
            
            # Get camera frame
            frame = self.camera.get_frame()
            if frame is None:
                print("Failed to get camera frame")
                continue
            
            # Detect bag
            print("Detecting bag...")
            px, py = self.detect_bag(frame)
            
            if px is None:
                print("No bag detected")
                continue
            
            # Convert to robot coordinates
            robot_x, robot_y = self.pixel_to_robot_coords(px, py, frame.shape)
            print(f"Bag at pixel ({px}, {py}) -> robot ({robot_x:.1f}, {robot_y:.1f})mm")
            
            # Execute pick
            try:
                episode_data = self.execute_pick_sequence(robot_x, robot_y)
                success = True
                self.success_count += 1
            except Exception as e:
                print(f"Pick failed: {e}")
                success = False
                episode_data = []
            
            self.total_attempts += 1
            
            # Record episode
            self.record_episode(episode_data, success)
            
            # Print statistics
            success_rate = self.success_count / self.total_attempts * 100
            print(f"Success rate: {self.success_count}/{self.total_attempts} ({success_rate:.1f}%)")
            
            # Wait before next episode
            time.sleep(2)
        
        print("\n" + "=" * 50)
        print(f"Completed {num_episodes} episodes")
        print(f"Final success rate: {self.success_count}/{self.total_attempts} ({success_rate:.1f}%)")
        
        if self.record_data:
            print(f"Dataset saved to: {self.dataset.root}")
    
    def teleoperation_mode(self):
        """Manual control with data recording"""
        print("\nTeleoperation Mode")
        print("=" * 50)
        print("Controls:")
        print("  SPACE - Detect and pick")
        print("  'r' - Reset to home")
        print("  's' - Start/stop recording")
        print("  'q' - Quit")
        
        recording = False
        episode_data = []
        
        cv2.namedWindow("LeRobot Bag Picker", cv2.WINDOW_AUTOSIZE)
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Display
            display = frame.copy()
            
            # Status
            status = "RECORDING" if recording else "READY"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow("LeRobot Bag Picker", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord(' '):
                # Detect and pick
                px, py = self.detect_bag(frame)
                if px is not None:
                    rx, ry = self.pixel_to_robot_coords(px, py, frame.shape)
                    episode_data = self.execute_pick_sequence(rx, ry)
                    
                    if recording:
                        self.record_episode(episode_data, success=True)
                
            elif key == ord('r'):
                # Reset
                if self.robot_type == "xarm":
                    self.robot.arm.set_position(x=0, y=300, z=300, wait=True)
                else:
                    self.robot.reset()
            
            elif key == ord('s'):
                recording = not recording
                if recording:
                    print("Started recording")
                    episode_data = []
                else:
                    print("Stopped recording")
        
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="xarm", help="Robot type")
    parser.add_argument("--mode", choices=["auto", "teleop"], default="auto", 
                       help="Operation mode")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of episodes for auto mode")
    parser.add_argument("--no-vlm", action="store_true", 
                       help="Use color detection instead of VLM")
    parser.add_argument("--no-record", action="store_true", 
                       help="Disable data recording")
    args = parser.parse_args()
    
    # Initialize system
    picker = LeRobotBagPicker(
        robot_type=args.robot,
        use_vlm=not args.no_vlm,
        record_data=not args.no_record
    )
    
    # Run selected mode
    if args.mode == "auto":
        picker.autonomous_mode(num_episodes=args.episodes)
    else:
        picker.teleoperation_mode()

if __name__ == "__main__":
    main()
