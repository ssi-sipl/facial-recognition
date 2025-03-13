import face_recognition
import pickle
import os
import numpy as np

dataset_path = "dataset"
encodings_file = "face_encodings.pkl"

known_encodings = []
known_names = []

for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    
    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}

with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"Training completed. Model saved as {encodings_file}.")
