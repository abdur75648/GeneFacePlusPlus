# Overlay generated video frame back on the top of original driver video (Larger video)

import cv2
import numpy as np
from tqdm import tqdm

# Define the file paths
original_video_path = 'BG_Segmented_Don_Georgevich_3_692_20s.mp4'
head_neck_video_path = 'headEnhancedResized_InpaintedOrigTorso.mp4'
output_video_path = 'Don_Final_Output_Video.mp4'

# Open both video files
original_video = cv2.VideoCapture(original_video_path)
head_neck_video = cv2.VideoCapture(head_neck_video_path)

# Get the properties of the head+neck video (width, height, and frame count)
head_width = int(head_neck_video.get(cv2.CAP_PROP_FRAME_WIDTH))
head_height = int(head_neck_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
head_fps = head_neck_video.get(cv2.CAP_PROP_FPS)
head_frame_count = int(head_neck_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the properties of the original video
original_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = original_video.get(cv2.CAP_PROP_FPS)

# Assert that the original video has the same fps as the head+neck video
assert original_fps == head_fps

# Create a VideoWriter object to save the final output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, head_fps, (original_width, original_height))

# Define the position to overlay the head+neck video (centered horizontally, top aligned)
x_offset = 418
y_offset = 0

# Loop through the frames of both videos
for i in tqdm(range(head_frame_count)):
    ret1, original_frame = original_video.read()
    ret2, head_neck_frame = head_neck_video.read()

    if not ret1 or not ret2:
        break  # Break the loop if we run out of frames

    # Paste the head+neck frame on top of the original frame
    original_frame[y_offset:y_offset + head_height, x_offset:x_offset + head_width] = head_neck_frame

    # Write the result to the output video
    out.write(original_frame)

# Release everything once the job is done
original_video.release()
head_neck_video.release()
out.release()

cv2.destroyAllWindows()
