import os
import cv2
import numpy as np
from tqdm import tqdm

from bg_segmenter import MediapipeSegmenter

def detect_faces_in_video(video_path, cascade_path, process_every_n_frames=100, detection_mode = "cv"):
    """Detects faces in a video, processes every Nth frame, and returns the bounding box values."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_path}")

    if detection_mode == "cv":
        cascade = cv2.CascadeClassifier(cascade_path)
    else:
        raise Exception("MediapipeSegmenter is not accuarte: Use OpenCV's HaarCascade instead")
        seg_model = MediapipeSegmenter()

    frame_count, ignored_frames = 0, 0
    x_min_list, y_min_list, x_max_list, y_max_list = [], [], [], []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while frame_count < total_frames:
            # Only process every Nth frame
            if frame_count % process_every_n_frames == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if detection_mode == "cv":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detections = cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)

                    if len(detections) == 1:
                        x, y, w, h = detections[0]
                        x_min_list.append(x)
                        y_min_list.append(y)
                        x_max_list.append(x + w)
                        y_max_list.append(y + h)
                    else:
                        ignored_frames += 1
                else:
                    raise Exception("MediapipeSegmenter is not accuarte: Use OpenCV's HaarCascade instead")
                    segmap = seg_model.segment_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mask = np.isin(segmap, [1, 3])
                    # Find the bounding box of the mask
                    y, x = np.where(mask)
                    x_min, y_min, x_max, y_max = np.min(x), np.min(y), np.max(x), np.max(y)
                    x_min_list.append(x_min)
                    y_min_list.append(y_min)
                    x_max_list.append(x_max)
                    y_max_list.append(y_max)

            frame_count += 1
            pbar.update(1)

    cap.release()
    
    if ignored_frames > frame_count // 2:
        raise Exception("Face detection failed for many frame")

    
    print(f"Processed {frame_count} frames, Ignored {ignored_frames} frames")
    
    return x_min_list, y_min_list, x_max_list, y_max_list


def calculate_bounding_box(x_min_list, y_min_list, x_max_list, y_max_list, increase_percentage=0.6):
    """Calculates a percentile-based bounding box, makes it square, and expands by a percentage."""
    def remove_outliers(data, lower_percentile, upper_percentile):
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        return [x for x in data if lower_bound <= x <= upper_bound]

    x_min_filtered = remove_outliers(x_min_list, 20, 80)
    y_min_filtered = remove_outliers(y_min_list, 20, 80)
    x_max_filtered = remove_outliers(x_max_list, 20, 80)
    y_max_filtered = remove_outliers(y_max_list, 20, 80)

    x_min = int(np.mean(x_min_filtered))
    y_min = int(np.mean(y_min_filtered))
    x_max = int(np.mean(x_max_filtered))
    y_max = int(np.mean(y_max_filtered))

    # Make the bounding box square (Here make the shorter side equal to the longer side)
    width, height = x_max - x_min, y_max - y_min
    if width > height:
        y_max = y_min + width
    else:
        x_max = x_min + height

    assert (x_max - x_min) == (y_max - y_min)  # Ensure it's square

    # Expand the bounding box by a percentage
    increase = int(increase_percentage * (x_max - x_min))//2
    return x_min - increase, y_min - increase, x_max + increase, y_max + increase


def adjust_bounding_box_to_frame(x_min, y_min, x_max, y_max, frame_width, frame_height):
    """Adjusts the bounding box to fit within the video frame dimensions."""
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

    # Ensure bounding box is square (Since it was already square before adjusting above, make the longer side equal to the shorter side)
    width = x_max - x_min
    height = y_max - y_min
    if width > height:
        x_max = x_min + height
    else:
        y_max = y_min + width
    
    # Round width and height to the nearest multiple of 10
    width = x_max - x_min
    new_width = (width // 10) * 10

    x_max = x_min + new_width
    y_max = y_min + new_width

    return x_min, y_min, x_max, y_max


def draw_bounding_box_on_video(video_path, output_path, bounding_box, max_duration=60):
    """Draws the calculated bounding box and the original box on all frames of the video."""
    x_min, y_min, x_max, y_max = bounding_box

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(fps * max_duration)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames_written = 0
    with tqdm(total=min(max_frames, total_frames), desc="Writing frames") as pbar:
        while True:
            if frames_written >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            # Draw the calculated rectangle (green) and the original rectangle (red)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            out.write(frame)
            pbar.update(1)
            frames_written += 1

    cap.release()
    out.release()
    print(f"Saved video to {output_path}")

def get_bounding_box_from_video(video_path, cascade_file='haarcascade_frontalface_default.xml', increase_percentage=0.6, process_every_n_frames=100):
    """Returns the final bounding box from the video."""
    cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)

    # Detect faces in video
    x_min_list, y_min_list, x_max_list, y_max_list = detect_faces_in_video(video_path, cascade_path, process_every_n_frames=process_every_n_frames)

    # Calculate bounding box
    x_min, y_min, x_max, y_max = calculate_bounding_box(x_min_list, y_min_list, x_max_list, y_max_list, increase_percentage)

    # Adjust bounding box to fit within video frame
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    x_min, y_min, x_max, y_max = adjust_bounding_box_to_frame(x_min, y_min, x_max, y_max, frame_width, frame_height)
    
    assert x_min >= 0 and y_min >= 0 and x_max <= frame_width and y_max <= frame_height and (x_max - x_min) == (y_max - y_min), "Invalid bounding box"
    
    print("Bounding box coordinates:", x_min, y_min, x_max, y_max)
    print("Bounding box dimensions:", x_max - x_min, y_max - y_min)
    
    return x_min, y_min, x_max, y_max


def main(video_path, output_path, cascade_file='haarcascade_frontalface_default.xml', increase_percentage=0.6, process_every_n_frames=100):
    """Main function to process the video, detect faces, and save the video with bounding boxes."""
    # Get the final bounding box from the video
    x_min, y_min, x_max, y_max = get_bounding_box_from_video(video_path, cascade_file, increase_percentage=increase_percentage, process_every_n_frames=process_every_n_frames)
    
    ### Uncomment to Ssve the obtained coordinates
    # with open(output_path.replace('.mp4', '.txt'), 'w') as f:
    #     f.write(f"{x_min} {y_min} {x_max} {y_max}")

    ### Uncomment to draw the bounding box on the video
    draw_bounding_box_on_video(video_path, output_path, (x_min, y_min, x_max, y_max), max_duration=2)
    print(f"Bounding box saved to {output_path}")


if __name__ == "__main__":
    main(video_path="VID20240927102042.mp4", output_path='VID20240927102042_output_cv.mp4', process_every_n_frames=100)