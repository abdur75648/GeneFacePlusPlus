import subprocess
from video_face_detection import get_bounding_box_from_video
import argparse

def crop_video(input_video, output_video, x1, y1, x2, y2, duration=None):
    # Calculate width and height from coordinates
    out_w = x2 - x1
    out_h = y2 - y1
    
    # Construct the ffmpeg command
    if duration is None:
        command = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f'crop={out_w}:{out_h}:{x1}:{y1}',
            '-c:a', 'copy',
            output_video
        ]
    else:
        assert float(duration) > 0, "Duration must be a positive number of seconds"
        command = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f'crop={out_w}:{out_h}:{x1}:{y1}',
            '-c:a', 'copy',
            '-t', str(duration),
            output_video
        ]
    
    # Execute the command
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Crop a video to a person's face bounding box.")
    parser.add_argument('--input_video', '-i', required=True, help="Path to the input video")
    parser.add_argument('--output_video', '-o', required=True, help="Path to the output video")
    parser.add_argument('--increase_percentage', '-p', type=float, default=1.0, help="Percentage to increase the bounding box (0 being the original size and 1 being 2x the size)")
    parser.add_argument('--no_trim', '-n', action='store_true', help="Do not save the trimmed video")
    args = parser.parse_args()

    # Read coordinates from the file
    x1, y1, x2, y2 = get_bounding_box_from_video(args.input_video,  increase_percentage=args.increase_percentage, process_every_n_frames=100)
    
    # Save the coordinates to a file
    with open(args.output_video.replace('.mp4', '.txt'), 'w') as f:
        f.write(f"{x1} {y1} {x2} {y2}")
    print(f"Bounding box coordinates saved as {args.output_video.replace('.mp4', '.txt')}")
    
    print("Cropping the video...")
    if not args.no_trim:
        crop_video(args.input_video, args.output_video.replace('.mp4', '_5s.mp4'), x1, y1, x2, y2, duration=5)
        print(f"Sampled cropped video (trimmed to 5s) saved as {args.output_video.replace('.mp4', '_5s.mp4')} - Make sure the face is correctly cropped!")
        exit("Exiting - Check the cropped video before processing the full video!")
    crop_video(args.input_video, args.output_video, x1, y1, x2, y2)
    print(f"Video cropped successfully! Final video saved as {args.output_video}")

if __name__ == "__main__":
    main()
