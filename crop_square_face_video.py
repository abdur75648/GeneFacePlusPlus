import subprocess
from video_face_detection import get_bounding_box_from_video
import argparse

def crop_video(input_video, output_video, x1, y1, x2, y2):
    # Calculate width and height from coordinates
    out_w = x2 - x1
    out_h = y2 - y1
    
    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'crop={out_w}:{out_h}:{x1}:{y1}',
        '-c:a', 'copy',
        output_video
    ]
    
    # Execute the command
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Crop a video to a person's face bounding box.")
    parser.add_argument('--input_video', '-i', required=True, help="Path to the input video")
    parser.add_argument('--output_video', '-o', required=True, help="Path to the output video")
    args = parser.parse_args()

    # Read coordinates from the file
    x1, y1, x2, y2 = get_bounding_box_from_video(args.input_video,  increase_percentage=0.6, process_every_n_frames=100)
    
    # Save the coordinates to a file
    with open(args.output_video.replace('.mp4', '.txt'), 'w') as f:
        f.write(f"{x1} {y1} {x2} {y2}")
    
    # Crop the video
    crop_video(args.input_video, args.output_video, x1, y1, x2, y2)
    print(f"Video cropped successfully and saved as {args.output_video}")

if __name__ == "__main__":
    main()
