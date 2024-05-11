import numpy as np
from keras_facenet import FaceNet
import cv2
import os
from mtcnn import MTCNN

# Initialize FaceNet model
print("Loading FaceNet model...")
facenet_model = FaceNet()
print("FaceNet model loaded successfully.")

# Initialize MTCNN detector
print("Loading MTCNN model...")
detector = MTCNN()
print("MTCNN model loaded successfully.")

def generate_face_embeddings(name, folder_path, output_folder_path, show_training=False):
    """
    Generate face embeddings from images in a folder and save them as a numpy file.

    Args:
        name (str): Name for the generated embeddings file.
        folder_path (str): Path to the folder containing images.
        output_folder_path (str): Path to the folder where the embeddings file will be saved.
        show_training (bool, optional): Flag to display training images during processing. Default is False.

    Returns:
        None
    """
 
    # Function to extract face embeddings from an image
    def extract_face_embeddings(image_path, count):
        """
        Extract face embeddings from an image using FaceNet model.

        Args:
            image_path (str): Path to the image file.
            count (int): Counter for displaying the progress.

        Returns:
            numpy.ndarray or None: Face embeddings if a face is detected, otherwise None.
        """
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
