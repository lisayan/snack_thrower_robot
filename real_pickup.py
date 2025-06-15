#!/usr/bin/env python3
"""
Real-world bag picking with SmolVLM2 and xArm
Simple, practical implementation for actual use
"""

import cv2
import torch
import numpy as np
from PIL import Image
import re
import time
from transformers import AutoProcessor, AutoModelForImageTextToText

# xArm imports - install with: pip install xArm-Python-SDK
try:
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    print("Warning: xArm SDK not found. Install with: pip install xArm-Python-SDK")
    XARM_AVAILABLE = False

# ----------------- Configuration -----------------
# xArm settings
XARM_IP = "192.168.1.100"  # UPDATE THIS to your xArm's IP address

# Camera settings
CAMERA_ID = 0  # 0 for default webcam, or camera index

# Pick positions (in mm for xArm)
HOME_POSITION = [200, 0, 300, 180, 0, 0]  # [x, y, z, roll, pitch, yaw]
APPROACH_HEIGHT = 200  # mm above table
PICK_HEIGHT = 30      # mm above table for grasping
DROP_POSITION = [-200, 0, 200, 180, 0, 0]  # Where to drop the bag

# Gripper settings (0-800 for xArm gripper)
GRIPPER_OPEN = 800
GRIPPER_CLOSE = 200

# Speed settings
MOVE_SPEED = 100     # mm/s
MOVE_ACCELERATION = 1000  # mm/s²

# Workspace calibration (you need to calibrate these!)
# Map camera pixels to robot coordinates
PIXELS_TO_MM_SCALE = 1.0  # Adjust based on your camera height
CAMERA_OFFSET_X = 0      # Offset between camera center and robot center
CAMERA_OFFSET_Y = 300    # Camera typically in front of robot base

# Model settings
MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cpu"  # Use "cuda" if available, "cpu" for compatibility
# ------------------------------------------------

class BagPicker:
    def __init__(self, use_real_robot=True):
        self.use_real_robot = use_real_robot and XARM_AVAILABLE
        
        # Initialize camera
        print("Initializing camera...")
        self.camera = cv2.VideoCapture(CAMERA_ID)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {CAMERA_ID}")
        
        # Get camera dimensions
        self.cam_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {self.cam_width}x{self.cam_height}")
        
        # Initialize robot
        if self.use_real_robot:
            print(f"Connecting to xArm at {XARM_IP}...")
            self.robot = XArmAPI(XARM_IP)
            self.robot.motion_enable(enable=True)
            self.robot.set_mode(0)
            self.robot.set_state(state=0)
            print("xArm connected!")
        else:
            print("Running without robot (simulation mode)")
            self.robot = None
            
        # Load VLM
        print("Loading SmolVLM2...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()
        print("Model loaded!")
        
    def detect_bag(self, frame):
        """Use SmolVLM2 to detect bag in frame"""
        # Convert to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Create prompt
        prompt = "Find the bag or package on the table. Return only the pixel coordinates of its center as 'x,y'. If no bag is visible, return '-1,-1'."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        try:
            # Process with model
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[img], return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=20
                )
            
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"VLM response: {response}")
            
            # Extract coordinates
            match = re.search(r'(-?\d+)\s*,\s*(-?\d+)', response)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                # Validate coordinates
                if 0 <= x < self.cam_width and 0 <= y < self.cam_height:
                    return x, y
                    
        except Exception as e:
            print(f"Detection error: {e}")
            
        return None, None
        
    def pixel_to_robot_coords(self, px, py):
        """Convert pixel coordinates to robot coordinates"""
        # Center coordinates
        cx = px - self.cam_width / 2
        cy = py - self.cam_height / 2
        
        # Scale to mm (you need to calibrate this!)
        # This is a simple linear mapping - improve with proper calibration
        robot_x = cx * PIXELS_TO_MM_SCALE + CAMERA_OFFSET_X
        robot_y = -cy * PIXELS_TO_MM_SCALE + CAMERA_OFFSET_Y  # Negative because image Y is flipped
        
        return robot_x, robot_y
        
    def move_to_position(self, x, y, z, roll=180, pitch=0, yaw=0):
        """Move robot to position"""
        if self.robot:
            print(f"Moving to: x={x:.1f}, y={y:.1f}, z={z:.1f}")
            code = self.robot.set_position(
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw=yaw,
                speed=MOVE_SPEED,
                mvacc=MOVE_ACCELERATION,
                wait=True
            )
            return code == 0
        else:
            print(f"[SIM] Move to: x={x:.1f}, y={y:.1f}, z={z:.1f}")
            time.sleep(1)
            return True
            
    def set_gripper(self, position):
        """Control gripper"""
        if self.robot:
            print(f"Setting gripper to: {position}")
            code = self.robot.set_gripper_position(position, wait=True)
            return code == 0
        else:
            state = "open" if position > 400 else "closed"
            print(f"[SIM] Gripper {state}")
            time.sleep(0.5)
            return True
            
    def go_home(self):
        """Move to home position"""
        return self.move_to_position(*HOME_POSITION)
        
    def pick_at_position(self, robot_x, robot_y):
        """Execute pick sequence at given position"""
        print(f"\nExecuting pick at ({robot_x:.1f}, {robot_y:.1f})")
        
        # 1. Move above the target
        if not self.move_to_position(robot_x, robot_y, APPROACH_HEIGHT):
            return False
            
        # 2. Open gripper
        if not self.set_gripper(GRIPPER_OPEN):
            return False
            
        # 3. Move down to pick height
        if not self.move_to_position(robot_x, robot_y, PICK_HEIGHT):
            return False
            
        # 4. Close gripper
        if not self.set_gripper(GRIPPER_CLOSE):
            return False
            
        # 5. Lift up
        if not self.move_to_position(robot_x, robot_y, APPROACH_HEIGHT):
            return False
            
        # 6. Move to drop position
        if not self.move_to_position(*DROP_POSITION):
            return False
            
        # 7. Open gripper to drop
        if not self.set_gripper(GRIPPER_OPEN):
            return False
            
        # 8. Return home
        return self.go_home()
        
    def run(self):
        """Main control loop"""
        print("\n" + "="*50)
        print("Bag Picking System Ready!")
        print("="*50)
        print("Controls:")
        print("  SPACE - Detect and pick")
        print("  'h' - Go home")
        print("  'c' - Calibration mode")
        print("  'q' - Quit")
        print("="*50 + "\n")
        
        # Start at home position
        self.go_home()
        
        calibration_mode = False
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break
                
            # Draw crosshair for calibration
            display = frame.copy()
            cv2.line(display, (self.cam_width//2, 0), (self.cam_width//2, self.cam_height), (0, 255, 0), 1)
            cv2.line(display, (0, self.cam_height//2), (self.cam_width, self.cam_height//2), (0, 255, 0), 1)
            
            if calibration_mode:
                cv2.putText(display, "CALIBRATION MODE - Click to test pixel mapping", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Bag Picker", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('h'):
                print("Going home...")
                self.go_home()
                
            elif key == ord('c'):
                calibration_mode = not calibration_mode
                print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
                
            elif key == ord(' '):  # Spacebar
                print("\nDetecting bag...")
                px, py = self.detect_bag(frame)
                
                if px is not None:
                    # Draw detection on screen
                    cv2.circle(display, (px, py), 10, (0, 255, 0), -1)
                    cv2.imshow("Bag Picker", display)
                    cv2.waitKey(500)  # Show detection briefly
                    
                    # Convert to robot coordinates
                    robot_x, robot_y = self.pixel_to_robot_coords(px, py)
                    print(f"Bag at pixel ({px}, {py}) -> robot ({robot_x:.1f}, {robot_y:.1f})")
                    
                    # Safety check
                    if abs(robot_x) > 400 or abs(robot_y) > 400:
                        print("WARNING: Position outside safe workspace!")
                        continue
                        
                    # Execute pick
                    if self.pick_at_position(robot_x, robot_y):
                        print("Pick complete!")
                    else:
                        print("Pick failed!")
                else:
                    print("No bag detected")
                    
            # Handle mouse clicks in calibration mode
            if calibration_mode:
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        rx, ry = self.pixel_to_robot_coords(x, y)
                        print(f"Pixel ({x}, {y}) -> Robot ({rx:.1f}, {ry:.1f})")
                        
                cv2.setMouseCallback("Bag Picker", mouse_callback)
                
    def cleanup(self):
        """Clean up resources"""
        self.camera.release()
        cv2.destroyAllWindows()
        if self.robot:
            self.robot.disconnect()

def main():
    picker = BagPicker(use_real_robot=True)
    
    try:
        picker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        picker.cleanup()

if __name__ == "__main__":
    main()
