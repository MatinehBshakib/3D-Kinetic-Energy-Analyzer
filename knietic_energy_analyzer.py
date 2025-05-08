import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
import time
import platform

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define landmark subsets from your original code
LANDMARK_SUBSETS = {
    "whole_body": list(range(33)),  # All landmarks
    "upper_body": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "lower_body": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    "right_arm": [11, 13, 15, 17, 19, 21],  # Right shoulder to right wrist and hand
    "left_arm": [12, 14, 16, 18, 20, 22],   # Left shoulder to left wrist and hand
    "right_leg": [23, 25, 27, 29, 31],      # Right hip to right ankle and foot
    "left_leg": [24, 26, 28, 30, 32],       # Left hip to left ankle and foot
}

# Your original KineticEnergyAnalyzer class
class KineticEnergyAnalyzer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.landmark_positions = [deque(maxlen=window_size) for _ in range(33)]
        self.velocities = [[] for _ in range(33)]
        self.last_timestamps = [None for _ in range(33)]
        self.kinetic_energies = {subset: 0.0 for subset in LANDMARK_SUBSETS.keys()}
        self.gesture_energies = []
        self.current_gesture_energies = {subset: 0.0 for subset in LANDMARK_SUBSETS.keys()}
        self.gesture_active = False
        # Assuming each landmark has equal mass (can be modified if needed)
        self.mass = 1.0
        
    def update_positions(self, landmarks, timestamp):
        """Update position history for each landmark"""
        for i, landmark in enumerate(landmarks.landmark[:33]):  # Using only the first 33 landmarks
            position = np.array([landmark.x, landmark.y, landmark.z])
            
            # If this is a valid position (not NaN)
            if not np.isnan(position).any():
                # Add position to history
                self.landmark_positions[i].append((position, timestamp))
                
                # Calculate velocity if we have previous position
                if self.last_timestamps[i] is not None:
                    time_delta = timestamp - self.last_timestamps[i]
                    if time_delta > 0:
                        prev_position = self.landmark_positions[i][-2][0]
                        velocity = (position - prev_position) / time_delta
                        self.velocities[i].append(velocity)
                        
                self.last_timestamps[i] = timestamp
    
    def compute_kinetic_energy(self):
        """Compute kinetic energy for all landmarks and subsets"""
        # Reset kinetic energies
        for subset in LANDMARK_SUBSETS.keys():
            self.kinetic_energies[subset] = 0.0
        
        # Compute kinetic energy for each landmark
        landmark_ke = [0.0] * 33
        
        for i in range(33):
            if len(self.velocities[i]) > 0:
                # Use the most recent velocity
                v = self.velocities[i][-1]
                # KE = 0.5 * m * |v|^2
                ke = 0.5 * self.mass * np.sum(v**2)
                landmark_ke[i] = ke
        
        # Accumulate kinetic energy for each subset
        for subset_name, indices in LANDMARK_SUBSETS.items():
            self.kinetic_energies[subset_name] = sum(landmark_ke[i] for i in indices if i < len(landmark_ke))
        
        return self.kinetic_energies
    
    def start_gesture(self):
        """Mark the beginning of a gesture"""
        self.gesture_active = True
        self.current_gesture_energies = {subset: 0.0 for subset in LANDMARK_SUBSETS.keys()}
        
    def update_gesture_energy(self):
        """Update the energy for the current gesture"""
        if self.gesture_active:
            for subset in LANDMARK_SUBSETS.keys():
                self.current_gesture_energies[subset] += self.kinetic_energies[subset]
    
    def end_gesture(self):
        """Mark the end of a gesture and store its energies"""
        if self.gesture_active:
            # Average the energies over the gesture duration
            num_frames = len(self.velocities[0]) if self.velocities[0] else 1
            avg_energies = {subset: energy / num_frames 
                           for subset, energy in self.current_gesture_energies.items()}
            
            self.gesture_energies.append(avg_energies)
            self.gesture_active = False
    
    def analyze_distribution(self, energies=None):
        """Analyze the distribution of kinetic energy among body parts"""
        if energies is None:
            energies = self.kinetic_energies
        
        # Skip whole_body as it's the sum of all others
        subsets = ["upper_body", "lower_body", "right_arm", "left_arm", "right_leg", "left_leg"]
        
        # If whole_body energy is very small, return early
        if energies["whole_body"] < 1e-6:
            return {subset: 0 for subset in subsets}, "No significant movement detected"
        
        # Calculate percentages
        percentages = {subset: (energies[subset] / energies["whole_body"]) * 100 
                      for subset in subsets}
        
        # Determine if any part is moving significantly more
        threshold = 40  # If a part has more than 40% of total energy
        max_subset = max(percentages.items(), key=lambda x: x[1])
        
        if max_subset[1] > threshold:
            message = f"The {max_subset[0].replace('_', ' ')} is moving a lot"
        else:
            message = "All limbs are moving relatively uniformly"
            
        return percentages, message
    
    def visualize_distribution(self, percentages):
        """Create a bar chart of energy distribution"""
        subsets = list(percentages.keys())
        values = list(percentages.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(subsets, values, color='skyblue')
        
        # Add percentage labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.xlabel('Body Part')
        plt.ylabel('Percentage of Total Kinetic Energy')
        plt.title('Distribution of Kinetic Energy Across Body Parts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()

# Modified analyze_webcam function for macOS compatibility
def analyze_webcam(duration=30, show_video=True, detection_threshold=0.5, gesture_detection_threshold=0.1):
    """
    Analyze body movement from webcam using MediaPipe - fixed for macOS compatibility
    
    Parameters:
    - duration: Duration to capture in seconds
    - show_video: Whether to show the video feed with pose overlay
    - detection_threshold: Confidence threshold for pose detection
    - gesture_detection_threshold: Energy threshold for gesture detection
    
    Returns:
    - analyzer: The KineticEnergyAnalyzer object with computed data
    - percentages: Distribution of energy across body parts
    - message: Analysis result message
    """
    print("Starting camera initialization...")
    
    # On macOS, we need to be extra careful with camera handling
    is_macos = platform.system() == "Darwin"
    if is_macos:
        print("Running on macOS - using macOS-specific camera handling")
    
    # Initialize camera - try a few times if needed
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"Camera initialization attempt {attempt+1}/{max_attempts}")
            cap = cv2.VideoCapture(0)
            
            # Important: Give camera time to initialize on macOS
            time.sleep(1)
            
            if not cap.isOpened():
                print("Failed to open camera - retrying...")
                cap.release()
                time.sleep(1)
                continue
                
            # Try reading a test frame to verify camera is working
            ret, test_frame = cap.read()
            if not ret:
                print("Camera opened but failed to read test frame - retrying...")
                cap.release()
                time.sleep(1)
                continue
                
            print(f"Camera initialized successfully! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            break
            
        except Exception as e:
            print(f"Error during camera initialization: {e}")
            if 'cap' in locals() and cap is not None:
                cap.release()
            time.sleep(1)
    else:
        # If we've exhausted all attempts
        print("ERROR: Could not initialize camera after multiple attempts.")
        print("Please check your camera permissions:")
        print("1. System Settings > Privacy & Security > Camera")
        print("2. Make sure your terminal/Python application has permission")
        print("3. Try closing any other applications that might be using the camera")
        return None, {}, "Camera initialization failed"
    
    # Setup MediaPipe Pose
    print("Initializing pose detection...")
    analyzer = KineticEnergyAnalyzer()
    
    with mp_pose.Pose(
        min_detection_confidence=detection_threshold,
        min_tracking_confidence=detection_threshold,
        model_complexity=1) as pose:
        
        start_time = time.time()
        frame_count = 0
        
        # For gesture detection
        current_gesture_energy = 0
        in_gesture = False
        
        print(f"Starting webcam analysis for {duration} seconds...")
        print("Press ESC to stop early")
        
        # Start a gesture
        analyzer.start_gesture()
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if we've reached the duration
            if elapsed_time > duration:
                break
                
            # Read frame
            success, image = cap.read()
            if not success:
                print("Failed to read frame - skipping")
                continue
            
            # Convert the BGR image to RGB and process it
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # If pose landmarks are detected
            if results.pose_landmarks:
                timestamp = current_time
                analyzer.update_positions(results.pose_landmarks, timestamp)
                energy = analyzer.compute_kinetic_energy()
                analyzer.update_gesture_energy()
                
                # Gesture detection
                whole_body_energy = energy["whole_body"]
                
                # Start a new gesture if energy exceeds threshold and we're not in a gesture
                if whole_body_energy > gesture_detection_threshold and not in_gesture:
                    print("Gesture started")
                    in_gesture = True
                
                # End gesture if energy drops below threshold and we're in a gesture
                if whole_body_energy < gesture_detection_threshold and in_gesture:
                    print("Gesture ended")
                    in_gesture = False
                
                # Analyze every 10 frames for display
                if frame_count % 10 == 0:
                    percentages, message = analyzer.analyze_distribution()
                    time_left = duration - elapsed_time
                    print(f"Time left: {time_left:.1f}s | {message}")
                    
                    if show_video:
                        # Draw the pose detection on the image
                        image.flags.writeable = True
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        
                        # Show energy info on the image
                        info_text = f"Time: {elapsed_time:.1f}s/{duration}s"
                        cv2.putText(image, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image, message, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Display the image
                        cv2.imshow('Body Movement Analysis', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            frame_count += 1
            
            # Process key presses - using waitKey(1) for better responsiveness
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Analysis stopped early")
                break
        
        # End any active gesture
        analyzer.end_gesture()
    
    # Properly release camera
    print("Analysis complete - releasing camera...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Final analysis
    if analyzer.gesture_energies:
        # Average the energies across all detected gestures
        avg_energies = {subset: 0 for subset in LANDMARK_SUBSETS.keys()}
        
        for gesture_energy in analyzer.gesture_energies:
            for subset, energy in gesture_energy.items():
                avg_energies[subset] += energy
        
        # Divide by number of gestures
        num_gestures = len(analyzer.gesture_energies)
        if num_gestures > 0:
            for subset in avg_energies:
                avg_energies[subset] /= num_gestures
        
        percentages, message = analyzer.analyze_distribution(avg_energies)
    else:
        percentages, message = analyzer.analyze_distribution()
    
    print("\nFinal Analysis:")
    print(message)
    print("\nEnergy Distribution (%):")
    for subset, percentage in percentages.items():
        if subset != "whole_body":  # Skip whole_body as it's always 100%
            print(f"{subset}: {percentage:.2f}%")
    
    # Visualize the distribution
    fig = analyzer.visualize_distribution(percentages)
    plt.show()
    
    return analyzer, percentages, message

# Simple test function for camera
def test_camera():
    """Simple test function to verify camera access"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        return False
    
    print(f"Successfully read frame with shape {frame.shape}")
    cap.release()
    return True

if __name__ == "__main__":
    # First run a simple test
    if test_camera():
        print("\nCamera test passed! Running main analysis...")
        # Run the analysis
        analyzer, percentages, message = analyze_webcam(duration=30)
    else:
        print("\nCamera test failed. Please check your camera permissions.")
