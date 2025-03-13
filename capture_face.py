import cv2
import os
import argparse

def capture_images(person_name, count=5):
    save_dir = f"dataset/{person_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Capturing {count} images for {person_name}... Press 'q' to quit.")
    img_count = 0
    
    while img_count < count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        cv2.imshow("Capture Face", frame)
        cv2.imwrite(f"{save_dir}/{img_count}.jpg", frame)
        img_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images for {person_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("person_name", help="Name of the person")
    parser.add_argument("--count", type=int, default=5, help="Number of images to capture")
    args = parser.parse_args()
    
    capture_images(args.person_name, args.count)
