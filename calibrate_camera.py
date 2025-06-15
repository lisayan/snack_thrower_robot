#!/usr/bin/env python3
"""
Simple Camera Calibration for Robot Workspace
Just calibration - nothing else!
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

class CameraCalibrator:
    def __init__(self, camera_index=2):
        self.camera_index = camera_index
        self.calibration_file = "camera_calibration.json"
        
        # Initialize camera
        print(f"Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.width}x{self.height}")
        
        # Calibration data
        self.calibration_points = []
        self.homography_matrix = None
        
    def calibrate_quick(self):
        """Quick 4-point calibration"""
        print("\n" + "="*60)
        print("QUICK CALIBRATION (4 points)")
        print("="*60)
        print("\nPlace markers at these exact robot positions:")
        print("  1. (-200, 200) mm  - Front Left")
        print("  2. ( 200, 200) mm  - Front Right")
        print("  3. ( 200, 400) mm  - Back Right")
        print("  4. (-200, 400) mm  - Back Left")
        print("\nThen click on each marker in order.")
        print("Press ESC to cancel.\n")
        
        # Reference positions
        reference_positions = [
            (-200, 200),  # Front Left
            (200, 200),   # Front Right
            (200, 400),   # Back Right
            (-200, 400),  # Back Left
        ]
        
        self.calibration_points = []
        current_point = 0
        
        # Mouse callback
        clicked_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_point
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point = (x, y)
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while current_point < 4:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Draw overlay
            overlay = frame.copy()
            
            # Title
            cv2.putText(overlay, "CAMERA CALIBRATION", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Draw collected points
            for i, (px, py, rx, ry) in enumerate(self.calibration_points):
                cv2.circle(overlay, (px, py), 10, (0, 255, 0), -1)
                cv2.putText(overlay, f"{i+1}", (px-5, py+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"({rx},{ry})", (px+15, py-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Current instruction
            if current_point < 4:
                rx, ry = reference_positions[current_point]
                point_names = ["Front Left", "Front Right", "Back Right", "Back Left"]
                
                instruction = f"Click on {point_names[current_point]} marker at robot position ({rx}, {ry})mm"
                cv2.putText(overlay, instruction, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw guide lines
                cv2.line(overlay, (self.width//2, 0), (self.width//2, self.height), (100, 100, 100), 1)
                cv2.line(overlay, (0, self.height//2), (self.width, self.height//2), (100, 100, 100), 1)
            
            cv2.imshow("Calibration", overlay)
            
            # Check for click
            if clicked_point is not None:
                px, py = clicked_point
                rx, ry = reference_positions[current_point]
                self.calibration_points.append((px, py, rx, ry))
                print(f"Point {current_point+1}: Clicked ({px}, {py}) -> Robot ({rx}, {ry})mm")
                current_point += 1
                clicked_point = None
            
            # Check for ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("Calibration cancelled")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        
        # Compute homography
        if self._compute_homography():
            print("\n✓ Calibration successful!")
            self._save_calibration()
            return True
        
        return False
    
    def calibrate_advanced(self):
        """Advanced calibration with more points"""
        print("\n" + "="*60)
        print("ADVANCED CALIBRATION")
        print("="*60)
        print("\nFor each point:")
        print("1. Move robot to the displayed position")
        print("2. Click where the robot gripper appears in camera")
        print("\nPress SPACE to skip a point, ESC to finish\n")
        
        # Generate grid of calibration positions
        positions = []
        for x in range(-300, 301, 150):
            for y in range(150, 451, 100):
                positions.append((x, y))
        
        self.calibration_points = []
        current_idx = 0
        clicked_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_point
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point = (x, y)
        
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        while current_idx < len(positions):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            overlay = frame.copy()
            
            # Draw all calibration points
            for px, py, rx, ry in self.calibration_points:
                cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
            
            # Current target
            target_x, target_y = positions[current_idx]
            
            cv2.putText(overlay, f"ADVANCED CALIBRATION - Point {current_idx+1}/{len(positions)}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(overlay, f"Move robot to: ({target_x}, {target_y})mm", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(overlay, "Then click on robot position in image", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.imshow("Calibration", overlay)
            
            # Check for click
            if clicked_point is not None:
                px, py = clicked_point
                self.calibration_points.append((px, py, target_x, target_y))
                print(f"Point {len(self.calibration_points)}: ({px}, {py}) -> ({target_x}, {target_y})")
                current_idx += 1
                clicked_point = None
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE - skip
                current_idx += 1
                print(f"Skipped point ({target_x}, {target_y})")
            elif key == 27:  # ESC - finish
                break
        
        cv2.destroyAllWindows()
        
        if len(self.calibration_points) >= 4:
            if self._compute_homography():
                print(f"\n✓ Calibration successful with {len(self.calibration_points)} points!")
                self._save_calibration()
                return True
        else:
            print("Need at least 4 points for calibration")
        
        return False
    
    def _compute_homography(self):
        """Compute transformation matrix"""
        if len(self.calibration_points) < 4:
            return False
        
        # Separate source (pixel) and destination (robot) points
        src_points = np.array([(p[0], p[1]) for p in self.calibration_points], dtype=np.float32)
        dst_points = np.array([(p[2], p[3]) for p in self.calibration_points], dtype=np.float32)
        
        # Compute homography
        self.homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        # Check quality
        if self.homography_matrix is not None:
            # Test accuracy
            total_error = 0
            for i, (px, py, rx, ry) in enumerate(self.calibration_points):
                # Transform pixel to robot
                point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(point, self.homography_matrix)
                pred_x, pred_y = transformed[0][0]
                
                error = np.sqrt((pred_x - rx)**2 + (pred_y - ry)**2)
                total_error += error
                
                if error > 10:  # More than 10mm error
                    print(f"Warning: Point {i+1} has {error:.1f}mm error")
            
            avg_error = total_error / len(self.calibration_points)
            print(f"\nAverage calibration error: {avg_error:.1f}mm")
            
            return True
        
        return False
    
    def _save_calibration(self):
        """Save calibration to file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "camera_index": self.camera_index,
            "resolution": [self.width, self.height],
            "num_points": len(self.calibration_points),
            "calibration_points": self.calibration_points,
            "homography_matrix": self.homography_matrix.tolist(),
            "method": "homography"
        }
        
        with open(self.calibration_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration saved to {self.calibration_file}")
    
    def test_calibration(self):
        """Test existing calibration"""
        # Load calibration
        if not Path(self.calibration_file).exists():
            print("No calibration file found!")
            return False
        
        with open(self.calibration_file, "r") as f:
            data = json.load(f)
        
        self.homography_matrix = np.array(data["homography_matrix"])
        self.calibration_points = data["calibration_points"]
        
        print("\n" + "="*60)
        print("CALIBRATION TEST")
        print("="*60)
        print("Click anywhere to see robot coordinates")
        print("Press 'g' to toggle grid")
        print("Press 'q' to quit\n")
        
        show_grid = True
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Transform pixel to robot
                point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(point, self.homography_matrix)
                robot_x, robot_y = transformed[0][0]
                print(f"Pixel ({x}, {y}) -> Robot ({robot_x:.1f}, {robot_y:.1f})mm")
        
        cv2.namedWindow("Test")
        cv2.setMouseCallback("Test", mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Draw calibration points
            for px, py, rx, ry in self.calibration_points:
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"({rx},{ry})", (px+10, py-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw grid
            if show_grid:
                # Draw robot coordinate grid
                for rx in range(-300, 301, 100):
                    for ry in range(100, 501, 100):
                        # Transform robot to pixel
                        inv_h = np.linalg.inv(self.homography_matrix)
                        point = np.array([[rx, ry]], dtype=np.float32).reshape(-1, 1, 2)
                        transformed = cv2.perspectiveTransform(point, inv_h)
                        px, py = int(transformed[0][0][0]), int(transformed[0][0][1])
                        
                        if 0 <= px < self.width and 0 <= py < self.height:
                            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)
                            if rx % 200 == 0 and ry % 200 == 0:
                                cv2.putText(frame, f"({rx},{ry})", (px+5, py-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            cv2.putText(frame, "CALIBRATION TEST - Click to test", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                show_grid = not show_grid
        
        cv2.destroyAllWindows()
        return True
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("Camera Calibration Tool")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2, help="Camera index")
    parser.add_argument("--mode", choices=["quick", "advanced", "test"], default="quick",
                       help="Calibration mode")
    args = parser.parse_args()
    
    try:
        calibrator = CameraCalibrator(camera_index=args.camera)
        
        if args.mode == "quick":
            success = calibrator.calibrate_quick()
            if success:
                print("\nWould you like to test the calibration? (y/n)")
                if input().lower() == 'y':
                    calibrator.test_calibration()
                    
        elif args.mode == "advanced":
            calibrator.calibrate_advanced()
            
        elif args.mode == "test":
            calibrator.test_calibration()
        
        calibrator.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
