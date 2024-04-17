import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1,
                                    min_face_detection_confidence=0.1,
                                    min_face_presence_confidence=0.1
                                    )
detector = vision.FaceLandmarker.create_from_options(options)


def run_mediapipe(image):
    # print(image.shape)    
    image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)


    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len (detection_result.face_landmarks) == 0:
        print('No face detected')
        return None
    
    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy
