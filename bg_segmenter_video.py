import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from multiprocessing import Pool
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter

class PersonBGSegmenter(MediapipeSegmenter):
    def __init__(self):
        super().__init__()
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

def process_video(video_path, output_video_path, seg_model, chunk_size=1000):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    assert fps == 25, "FPS of the video is not 25"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if os.cpu_count() > 2:
        cpu_count = os.cpu_count() - 2
    else:
        cpu_count = 1
    print(f"Using {cpu_count} cores for parallel processing")

    for start_frame in range(0, frame_count, chunk_size):
        print(f"Processing frames {start_frame} to {min(start_frame + chunk_size, frame_count)}")
        end_frame = min(start_frame + chunk_size, frame_count)
        frame_info_list = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in tqdm(range(start_frame, end_frame), desc="Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frame_info_list.append((frame, frame_num, seg_model))

        with Pool(cpu_count) as p:
            for frame_num, processed_frame in tqdm(p.imap(process_frame, frame_info_list), total=len(frame_info_list)):
                out_video.write(processed_frame)
        
        del frame_info_list

    cap.release()
    out_video.release()
    print("Segmentation Done - Saved!")

if __name__ == '__main__':
    video_path = "VID20240927102042.mp4"
    output_video_path = "BG_Segmented_VID20240927102042.mp4"
    seg_model = PersonBGSegmenter()

    process_video(video_path, output_video_path, seg_model)
