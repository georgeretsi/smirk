import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process images/videos with MediaPipe.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations')
parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use for processing')
args = parser.parse_args()


# Function to process an image
def process_image(image_file, output_file, vis_file, face_detector):
    image = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load and process the image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = face_detector.detect(mp_image)

    # Check for detected faces
    if not detection_result.face_landmarks:
        return

    # Extract face landmarks
    landmarks = detection_result.face_landmarks[0]
    landmarks_np = np.array([[landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z] for landmark in landmarks])

    # Save landmarks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, landmarks_np)

    # Save visualization if required
    if vis_file:
        for landmark in landmarks_np:
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
        os.makedirs(os.path.dirname(vis_file), exist_ok=True)
        cv2.imwrite(vis_file, image)

# Function to process a video
def process_video(video_file, output_file, vis_file, face_detector):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_landmarks = []

    if vis_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vis_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = face_detector.detect(mp_frame)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            landmarks_np = np.array([[landmark.x * mp_frame.width, landmark.y * mp_frame.height, landmark.z] for landmark in landmarks])
            frame_landmarks.append(landmarks_np)

            if vis_file:
                for landmark in landmarks_np:
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)

        if vis_file:
            out.write(frame)

    cap.release()
    if vis_file:
        out.release()

    # Save video landmarks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, np.array(frame_landmarks))


# Function to process a file
def process_file(root, file_name, face_detector):
    input_path = os.path.join(root, file_name)
    rel_path = os.path.relpath(input_path, args.input_dir)
    output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.npy')
    vis_path = os.path.join(args.vis_dir, rel_path) if args.vis_dir else None

    if file_name.lower().endswith(('.jpg', '.png')):
        process_image(input_path, output_path, vis_path, face_detector)
    elif file_name.lower().endswith(('.mp4', '.avi')):
        process_video(input_path, output_path, vis_path, face_detector)

# Main processing function
def process_sample(args):
    root, file_name = args
    face_detector_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='assets/face_landmarker.task'),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_detector_options)

    process_file(root, file_name, face_detector)

if __name__ == '__main__':
    all_files = []

    for root, _, files in os.walk(args.input_dir):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.png', '.mp4', '.avi')):
                all_files.append((root, file_name))

    with Pool(args.num_processes) as pool:
        list(tqdm(pool.imap(process_sample, all_files), total=len(all_files)))