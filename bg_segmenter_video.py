import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from multiprocessing import Pool
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

class MediapipeSegmenter:
    def __init__(self):
        model_path = 'data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("Downloading segmenter model from Mediapipe...")
            os.system(f"wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
            os.system(f"mv selfie_multiclass_256x256.tflite {model_path}")
            print("Download success")

        base_options = BaseOptions(model_asset_path=model_path)
        self.options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )

    def segment_image(self, img):
        segmenter = vision.ImageSegmenter.create_from_options(self.options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        out = segmenter.segment(mp_image)
        return out.category_mask.numpy_view().copy()

    def apply_green_screen(self, img, segmap):
        green_background = np.zeros_like(img)
        green_background[..., 1] = 255  # Set green channel to 255

        # Assuming the category_mask contains 0 for background
        mask = segmap > 0
        green_screen_img = np.where(mask[:, :, None], img, green_background)
        return green_screen_img

def process_frame(frame_info):
    frame, frame_num, seg_model = frame_info
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    segmap = seg_model.segment_image(img)
    green_screen_img = seg_model.apply_green_screen(img, segmap)
    return frame_num, cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR)

def process_video(video_path, output_video_path, seg_model):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    cpu_count = 2 # os.cpu_count()
    print(f"Using {cpu_count} cores for parallel processing")

    frame_info_list = []

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_info_list.append((frame, frame_num, seg_model))

    cap.release()

    with Pool(cpu_count) as p:
        for frame_num, processed_frame in tqdm(p.imap(process_frame, frame_info_list), total=frame_count):
            out_video.write(processed_frame)

    out_video.release()

if __name__ == '__main__':
    video_path = "Don_Georgevich_3_692_20s_25fps.mp4"
    output_video_path = "BG_Segmented_Don_Georgevich_3_692_20s.mp4"
    seg_model = MediapipeSegmenter()

    process_video(video_path, output_video_path, seg_model)
