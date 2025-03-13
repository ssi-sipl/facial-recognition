# File: capture_face.py
# Purpose: Capture reference images for facial recognition

import cv2
import os
import time
import argparse

def capture_reference_images(person_name, num_images=5):
    # Create directory for person if it doesn't exist
    save_dir = f"faces/{person_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the Pi camera
    camera = cv2.VideoCapture(0)  # Use 0 for first camera
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Capturing {num_images} images for {person_name}")
    print("Look at the camera and change your expression slightly between captures")
    
    # Wait for camera to initialize
    time.sleep(2)
    
    for i in range(num_images):
        # Countdown
        for j in range(3, 0, -1):
            print(f"{j}...")
            time.sleep(1)
            
        # Capture frame
        ret, frame = camera.read()
        
        if not ret:
            print("Failed to capture image")
            continue
            
        # Save the image
        image_path = os.path.join(save_dir, f"{person_name}_{i+1}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image {i+1}/{num_images} saved to {image_path}")
        
        # Short pause between captures
        time.sleep(1)
    
    camera.release()
    print(f"Finished capturing reference images for {person_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture reference images for facial recognition")
    parser.add_argument("name", help="Name of the person to capture")
    parser.add_argument("--count", type=int, default=5, help="Number of images to capture")
    args = parser.parse_args()
    
    capture_reference_images(args.name, args.count)

# File: train_model.py
# Purpose: Process reference images and create face embeddings

import cv2
import os
import numpy as np
import pickle
import face_recognition
import argparse

def train_face_recognition_model():
    print("Training face recognition model...")
    
    # Dictionary to store face encodings
    known_face_encodings = []
    known_face_names = []
    
    # Get all subdirectories in the faces directory
    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        print(f"Error: {faces_dir} directory not found")
        return
    
    person_dirs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    
    if not person_dirs:
        print("No person directories found. Capture faces first.")
        return
    
    # Process each person's directory
    for person_name in person_dirs:
        person_dir = os.path.join(faces_dir, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No images found for {person_name}")
            continue
        
        print(f"Processing {len(image_files)} images for {person_name}")
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                print(f"No face found in {image_path}")
                continue
            
            # Get face encodings (using the first face found)
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Add to our lists
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)
    
    # Save the face encodings and names
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"Model trained with {len(known_face_names)} face encodings for {len(set(known_face_names))} people")

if __name__ == "__main__":
    train_face_recognition_model()

# File: recognize_faces.py
# Purpose: Real-time face recognition from camera feed

import cv2
import pickle
import face_recognition
import numpy as np
import time

def recognize_faces():
    # Load the trained face encodings
    try:
        with open("face_encodings.pkl", "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    except FileNotFoundError:
        print("Error: Face encodings file not found. Run train_model.py first.")
        return
    
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set a smaller frame size for faster processing
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting face recognition. Press 'q' to quit.")
    
    # Process frames in real-time
    process_this_frame = True
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Failed to capture frame")
            break
        
        # Only process every other frame to save time
        if process_this_frame:
            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR to RGB (face_recognition uses RGB)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # Compare face with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()