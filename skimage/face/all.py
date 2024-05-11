import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from mtcnn import MTCNN
# from ultralytics import YOLO

# Initialize FaceNet model
print("Loading FaceNet Model...")
facenet_model = FaceNet()
print("FaceNet model loaded successfully.")

# Initialize MTCNN detector
detector = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_recognition(cropped_image, embedding_files, threshold=0.65):
    """
    Recognize faces in the cropped image and return the recognized person along with the similarity score.

    Args:
        cropped_image (numpy.ndarray): Cropped face region from the input image.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.

    Returns:
        tuple: Recognized person name and the maximum similarity score.
    """
    loaded_face_embeddings = [np.load(file) for file in embedding_files]

    # Find embeddings of the face in the test image
    test_face_embeddings = facenet_model.embeddings([np.array(cropped_image)])

    # Compare the test face embeddings with the loaded embeddings
    similarity_scores = [np.dot(loaded_face_embeddings[i], test_face_embeddings.T) for i in range(len(loaded_face_embeddings))]

    # Check if the test face belongs to any person in the database
    max_score = np.max(similarity_scores)
    if max_score > threshold:
        label = np.argmax(similarity_scores)
        base_name = os.path.basename(embedding_files[label])
        name_without_suffix = base_name.replace("_embeddings.npy", "")
        recognized_person = name_without_suffix.replace("_", " ").title()
    else:
        recognized_person = "Unknown"

    return recognized_person, max_score

def generate_face_embeddings(name, folder_path, output_folder_path, show_training=False):
    """
    Generate face embeddings from images in the given folder and save them to output folder.

    Args:
        name (str): Name for the output embeddings file.
        folder_path (str): Path to the folder containing images.
        output_folder_path (str): Path to the output folder where embeddings will be saved.
        show_training (bool, optional): Flag to display training images. Default is False.
    """
    load_mtcnn_model()

    # Function to extract face embeddings from an image
    def extract_face_embeddings(image_path, count):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face in the image
        results = detector.detect_faces(image)

        if results:
            x, y, w, h = results[0]['box']
            x, y, w, h = int(x), int(y), int(w), int(h)
            cropped_image = image[y:y + h, x:x + w]

            resized_image = cv2.resize(cropped_image, (160, 160))

            if show_training:
                cv2.imshow(f"{image_path}", resized_image)
                cv2.waitKey(1)

            # Find embeddings of the face in the image
            face_embeddings = facenet_model.embeddings([np.array(resized_image)])
            print(str(count) + " Completed")

            return face_embeddings.flatten()  # Flatten to make it a 1D array
        else:
            return None

    # List to store face embeddings
    all_embeddings = []

    # Iterate through each image in the folder
    count = 0
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(folder_path, image_file)
            count += 1

            # Extract face embeddings from the image
            face_embeddings = extract_face_embeddings(image_path, count)

            if face_embeddings is not None:
                # Add face embeddings to the list
                all_embeddings.append(face_embeddings)

    # Combine the embeddings into a single embedding vector
    combined_embedding = np.mean(all_embeddings, axis=0)

    # Save the numpy file
    name = name.replace(" ", "_")
    file_name = os.path.join(output_folder_path, f'{name}_embeddings.npy')
    np.save(file_name, combined_embedding)
    print(f"Embeddings generated and saved successfully as {file_name}")
    cv2.destroyAllWindows()

def identify_faces_in_image(test_image, embedding_files, threshold=0.65, show_frame=True):
    """
    Identify faces in a single image using MTCNN model and recognize them using FaceNet.

    Args:
        test_image (numpy.ndarray): Input image.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.
        show_frame (bool, optional): Flag to display the processed image with recognized faces. Default is True.

    Returns:
        dict: Dictionary containing recognized faces and their similarity scores.
    """
    global facenet_model

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    load_mtcnn_model()
    results = detector.detect_faces(image_np_original)
    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores

    if results:
        for r in results:
            x, y, width, height = r['box']
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + width), int(y + height)
            if show_frame:
                cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

            # Crop the face region
            face_roi = cv2.resize(test_image[ymin:ymax, xmin:xmax], (160, 160))

            # Perform face recognition using the face_recognition function
            recognized_person, max_score = face_recognition(face_roi, embedding_files, threshold=threshold)

            if recognized_person == "Unknown":
                continue
            else:
                recognized_faces[recognized_person] = max_score  # Add to recognized_faces dictionary

            if show_frame:
                cv2.putText(image_np_original, f"{recognized_person} ({max_score:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        image_np_original = cv2.cvtColor(image_np_original, cv2.COLOR_BGR2RGB)

        if show_frame:
            cv2.imshow("Detected and Recognized Faces", image_np_original)
            print("Press any key to continue.")
            cv2.waitKey(0)
    else:
        print("No faces found")

    return recognized_faces

def identify_faces_in_video(test_image, embedding_files, yolo_model, threshold=0.65, show_frame=True):
    """
    Identify faces in a video stream using YOLO model and recognize them using FaceNet.

    Args:
        test_image (numpy.ndarray): Input video frame.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.
        show_frame (bool, optional): Flag to display the processed frame with recognized faces. Default is True.

    Returns:
        dict: Dictionary containing recognized faces and their similarity scores.
    """

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    image_np_original = np.array(test_image)

    results = yolo_model.predict(test_image, show=False, stream=False)
    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores

    if results:
        for r in results:
            for box, score, class_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                if show_frame:
                    cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

                # Crop the face region
                face_roi = cv2.resize(test_image[ymin:ymax, xmin:xmax], (160, 160))

                # Perform face recognition using the face_recognition function
                recognized_person, max_score = face_recognition(face_roi, embedding_files, threshold=threshold)

                if recognized_person == "Unknown":
                    continue
                else:
                    recognized_faces[recognized_person] = max_score  # Add to recognized_faces dictionary

                if show_frame:
                    cv2.putText(image_np_original, f"{recognized_person} ({max_score:.2f})", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        image_np_original = cv2.cvtColor(image_np_original, cv2.COLOR_BGR2RGB)
        if show_frame:
            cv2.imshow("Frame", image_np_original)
    else:
        print("No faces found")

    return recognized_faces


def simple_identify_faces_in_video(test_image, embedding_files, threshold=0.65, show_frame=True):
    """
    Identify faces in the input image using a simple method and return recognized faces along with their similarity scores.

    Args:
        test_image (numpy.ndarray): Input image.
        embedding_files (list): List of paths to the embedding files.
        threshold (float, optional): Threshold value for similarity score. Default is 0.65.
        show_frame (bool, optional): Flag to display the processed image with recognized faces. Default is True.

    Returns:
        dict: Dictionary containing recognized faces and their similarity scores.
    """
    global facenet_model

    # Convert the image to grayscale
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    recognized_faces = {}  # Dictionary to store recognized faces and their similarity scores
    image_np_original = np.array(test_image)

    for (x, y, w, h) in faces:
        xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)
        if show_frame:
            cv2.rectangle(image_np_original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

        face_roi = cv2.resize(test_image[ymin:ymax, xmin:xmax], (160, 160))

        # Perform face recognition using the face_recognition function
        recognized_person, max_score = face_recognition(face_roi, embedding_files, threshold=threshold)

        if recognized_person == "Unknown":
            continue
        else:
            recognized_faces[recognized_person] = max_score  # Add to recognized_faces dictionary

        if show_frame:
            cv2.putText(image_np_original, f"{recognized_person} ({max_score:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the result
    if show_frame:
        cv2.imshow('Detected and Recognized Faces', image_np_original)

    return recognized_faces
