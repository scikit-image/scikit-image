from setuptools import setup, find_packages

setup(
    name='myf_face_recognition',
    version='0.14',
    packages=find_packages(include=["myf_face_recognition"]),
    include_package_data=True,
    package_data={'myf_face_recognition.models': ['yolov8n-face.pt']},
    install_requires=[
        'numpy',
        'keras-facenet',
        'opencv-python',
        'mtcnn',
        'ultralytics',
        'tensorflow'
        
    ],
    entry_points={
        'console_scripts': [
            'myf_face_recognition=myf_face_recognition.__main__:main'
        ]
    },
    author='Virendrakumar Bind',
    author_email='finalyearprojectetc@gmail.com',
    description='An easy-to-use yet powerful Python library for face recognition tasks.',
    license='MIT',
    keywords='face-recognition',
    url='https://github.com/VirNotFound/myf_face_recognition_updated',
)
