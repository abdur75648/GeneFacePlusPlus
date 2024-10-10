# Overlay generated video frame back on the top of original driver video (Larger video)

import cv2
import numpy as np
from tqdm import tqdm
import moviepy.editor as mp

# Define the file paths
original_video_path = 'BG_Segmented_VID20240927102042.mp4'
head_neck_video_path = 'Girish1_head_best.sr_4x.mp4'
output_video_path = 'Girish1_Final_Output_Video.mp4'

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
assert original_fps == head_fps, "FPS of the two videos do not match. The original_fps is {} and the head_fps is {}".format(original_fps, head_fps)

# Create a VideoWriter object to save the final output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, head_fps, (original_width, original_height))

# Read offset from the file as x1 y1 x2 y2 and resize the head video to the size of x2-x1 y2-y1, then paste it at x1 y1
crop_file = "data/raw/videos/Girish1.txt"
print("Reading coordinates from", crop_file)
with open(crop_file, 'r') as f:
    x1, y1, x2, y2 = map(int, f.readline().strip().split())
    x_offset = x1
    y_offset = y1
    head_width = x2 - x1
    head_height = y2 - y1
    assert head_width==head_height
print("Coordinates read from", crop_file, "are x1, y1, x2, y2:", x1, y1, x2, y2)

# Loop through the frames of both videos
for i in tqdm(range(head_frame_count)):
    ret1, original_frame = original_video.read()
    ret2, head_neck_frame = head_neck_video.read()

    if not ret1 or not ret2:
        break  # Break the loop if we run out of frames
    
    # Resize the head+neck frame to the desired size
    head_neck_frame = cv2.resize(head_neck_frame, (head_width, head_height))

    # Paste the head+neck frame on top of the original frame
    original_frame[y_offset:y_offset + head_height, x_offset:x_offset + head_width] = head_neck_frame

    # Write the result to the output video
    out.write(original_frame)

# Release everything once the job is done
original_video.release()
head_neck_video.release()
out.release()
cv2.destroyAllWindows()
print("Output video saved to", output_video_path)

print("Adding audio to the output video from ", head_neck_video_path)
head_neck_clip = mp.VideoFileClip(head_neck_video_path)
audio = head_neck_clip.audio
if audio is None:
    raise ValueError("Failed to extract audio from ", head_neck_video_path)
final_clip = mp.VideoFileClip(output_video_path)
final_clip = final_clip.set_audio(audio)
final_clip.write_videofile(output_video_path.replace(".mp4", "_with_audio.mp4"), codec="libx264", audio_codec="aac")
final_clip.close()
head_neck_clip.close()
print("Audio added to the output video and saved as", output_video_path.replace(".mp4", "_with_audio.mp4"))