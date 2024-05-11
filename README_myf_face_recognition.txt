myf_face_recognition

myf_face_recognition is an easy-to-use yet powerful Python library that integrates the capabilities of MTCNN, YOLO, and FaceNet for efficient face recognition tasks.
It can find a person in a picture containing a bunch of people as well.
The library generally works great even in scenarios containing harsh lighting conditions, partial face occlusions etc if trained with suitably large and diverse dataset.


Libraries required-

pip install numpy                        1.26.3
pip install keras-facenet                0.3.2
pip install opencv-python                4.7.0.72
pip install mtcnn                        0.1.1
pip install ultralytics                  8.0.235


Python (3.10.0)


Folder Structure:

-root
    -data
        Image.jpg
        Video.mp4
        .
        .
        .
    -embeddings
        Tony_Stark_embeddings.npy
        .
        .
        .
    -myf_face_recognition
        all.py
        embeddings.py
        images.py
        simple_video.py
        videos.py

    README.txt
    yolov8n-face.pt
    test_script.py (Create your python scripts here to use this library)

Don't change the myf_face_recognition folder Structure,
and create your scripts in the root directory directly and not in some folders.



Functions

The library consists of mainly three functions:

Generate Face Embeddings
Identify Faces in Images
Identify Faces in Videos



1. Generate Face Embeddings

What are Face Embeddings?
Face embeddings are vector representations of facial features extracted from images. These vectors encode unique characteristics of faces in a high-dimensional space, enabling efficient comparison and recognition of faces. In simple words, face embeddings are mathematical representation of a face, it is unique for each faces.

Usage-

generate_face_embeddings(name, folder_path, output_folder_path, show_training=False)

Args-
name: Name of the person.
folder_path: Path to the folder containing images.
output_folder_path: Path to save the generated embeddings.
show_training: Whether to display the training process (default: False).

Return-
None

The allowed extensions for images in the folder are .jpg .jpeg .png .webp only.
The images can be named however required, it does not matter the training process.
The images MUST contain only one face, and that to, of the subject only, group pictures will NOT work, they will mess up the training.
For descent results, it is neccessary to have about 20 images of the same subject, from various different angles of his face. The more the training images in different conditions and backgrounds, the better, accurate and stable the results.
The embeddings generated will be saved in the folder embeddings with the name, Person_embeddings.npy

The output_folder_path will be "embeddings".

It is recommended to keep show_training=True, for verifying if the training process is going correctly or not, but it consumes more memory.

Here is a dummy script for face embedding generation.
{
    from myf_face_recognition.embeddings import generate_face_embeddings

    name = "Tony Stark"
    folder_path = 'Tony Stark'
    output_folder_path = 'embeddings'
    generate_face_embeddings(name, folder_path, output_folder_path, show_training=True)

}



2. Identify Faces in Images

Usage-

identify_faces_in_image(test_image, embedding_files, threshold=0.65, show_frame=True)

Args-
test_image: Pass the image in which you want to search the person, this can be done using cv2 library or some other libraries.
embedding_files: This is a list of person's embeddings for whome you want to search in the image.
threshold: This is the trigger knob, adjust this to fine tune the results (default: 0.65).
show_frame: Whether to show the image and highlight his/her location (default: True).

Return-
recognized_faces: This function returns a dictionary containing the name of the persons as the keys and the accuracy as their values.

Here is a dummy script for face recognition in an image.
{
    from myf_face_recognition.images import identify_faces_in_image
    import cv2

    # Set a threshold for similarity
    threshold = 0.7

    # List of embedding files
    embedding_files = [
        "embeddings/Tony_Stark_embeddings.npy",
        # Add more paths as needed
    ]

    test_image = cv2.imread("test_tony.jpeg")
    recognized_faces = identify_faces_in_image(test_image, embedding_files, show_frame=True)
    print(recognized_faces)
}


3. Identify Faces in Videos

This is very similar to that of the face recognition in image, essentially this is the same thing but in a loop, we pass each frame of the video to the function and it gives us the results.

Usage-

identify_faces_in_video(test_image, embedding_files, yolo_model, threshold=0.65, show_frame=True)

Args-
test_image: Pass each frame of the video in which you want to search the person, this can be done using cv2 library using a loop or some other libraries.
embedding_files: This is a list of person's embeddings for whome you want to search in the image.
threshold: This is the trigger knob, adjust this to fine tune the results (default: 0.65).
yolo_model: You will need a pre-trained face detection model or custom trained model for face detection, pass its path here.
show_frame: Whether to show the frames and highlight his/her location (default: True).

Return-
recognized_faces: This function returns a dictionary containing the name of the persons as the keys and the accuracy as their values.

Here is a dummy script for face recognition in a video.
{
    from myf_face_recognition.video import identify_faces_in_video
    import cv2
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n-face.pt')

    # Set a threshold for similarity
    threshold = 0.7

    # List of embedding files
    embedding_files = [
        "embeddings/Tony_Stark_embeddings.npy",
        # Add more paths as needed
    ]

    video_path = "Tony.mp4"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't Load")
            break

        results = identify_faces_in_video(frame, embedding_files, yolo_model)
        print(results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release() 
    cv2.destroyAllWindows()

}



This function uses YOLO for face detection and then FaceNet for face recognition, due to YOLO it becomes a little slower, so instead another option can be considered but the accuracy drops significantly.

Here is the implementation.
{
    from myf_face_recognition.simple_video import simple_identify_faces_in_video
    import cv2

    # Set a threshold for similarity
    threshold = 0.7

    # List of embedding files
    embedding_files = [
        "embeddings/Tony_Stark_embeddings.npy",
        # Add more paths as needed
    ]

    video_path = "Tony.mp4"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't Load")
            break

        results = simple_identify_faces_in_video(frame, embedding_files)
        print(results)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cv2.destroyAllWindows()

}

This approach uses Haar Cascade technique for face detection which is must faster than YOLO but the accuracy is also really less.


At last there is an all.py file which contains all of the above discussed function in it, but it is recommended to use dedicated import commands and not this all module for skipping the unnecessary loading time of models that are not required in the script.
But for simplicity, one can just import this commands and get the things done without much problems.

from myf_face_recognition.embeddings import generate_face_embeddings
from myf_face_recognition.images import identify_faces_in_image
from myf_face_recognition.video import identify_faces_in_video
from myf_face_recognition.simple_video import simple_identify_faces_in_video

These four import statements are combined in the below one single command

from myf_face_recognition.all import generate_face_embeddings, identify_faces_in_image, identify_faces_in_video, simple_identify_faces_in_video



Improvements-
The performance of the library can be Improved by doing the following things
- Using a dedicated and powerful GPU.
- Using a powerful CPU.
- Using a more accurate YOLO face models.
- Loading only important models.
