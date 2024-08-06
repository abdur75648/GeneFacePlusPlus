#!/bin/bash
set -e

# Capture start time
START_TIME=$(date +%s)

# Assuming video is saved in data/raw/videos/${VIDEO_ID}.mp4
export VIDEO_ID="AbdurHD"

# Step 0: Standardize video resolution to 1024x1024 and FPS to 25
echo "Step 0.1: Standardizing video resolution and FPS..."
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=1024:h=1024 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_1024.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_1024.mp4 data/raw/videos/${VIDEO_ID}.mp4

# Step 0.2: Creating config files
echo "Step 0.2: Creating config files..."
# Copy all .yaml files from egs/datasets/Custom/ to egs/datasets/${VIDEO_ID}/
mkdir -p egs/datasets/${VIDEO_ID}
cp egs/datasets/Custom/*.yaml egs/datasets/${VIDEO_ID}/
# change the ```video_id``` in egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso.yaml from ```Custom``` to your ```VIDEO_ID```
sed -i "s/video_id: Custom/video_id: $VIDEO_ID/" "egs/datasets/$VIDEO_ID/lm3d_radnerf_torso.yaml" 
# change the ```video_id``` in egs/datasets/${VIDEO_ID}/lm3d_radnerf.yaml from ```Custom``` to your ```VIDEO_ID```
sed -i "s/video_id: Custom/video_id: $VIDEO_ID/" "egs/datasets/$VIDEO_ID/lm3d_radnerf.yaml"
# Change the ```head_model_dir``` in ```lm3d_radnerf_torso.yaml``` as per your ```VIDEO_ID``` [Format is checkpoints/$VIDEO_ID/lm3d_radnerf]
sed -i "s/head_model_dir: checkpoints\/Custom\/lm3d_radnerf/head_model_dir: checkpoints\/$VIDEO_ID\/lm3d_radnerf/" "egs/datasets/$VIDEO_ID/lm3d_radnerf_torso.yaml"

# Step 1: Extract audio features
echo "Step 1: Extracting audio features..."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=./
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID} # Will be saved at data/processed/videos/${VIDEO_ID}/aud_hubert.npy
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID} # Will be saved at data/processed/videos/${VIDEO_ID}/aud_mel_f0.npy

# Step 2: Extract images
echo "Step 2: Extracting images..."
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=1024:h=1024 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
# Following subfolders are created inside data/processed/videos/${VIDEO_ID}/
# - bg_imgs, com_imgs, gt_imgs, head_imgs, inpaint_torso_imgs, person_imgs, segmaps, torso_imgs
echo "Images extracted and saved in subfolders: bg_imgs, com_imgs, gt_imgs, head_imgs, inpaint_torso_imgs, person_imgs, segmaps, torso_imgs"

# Step 3: Extract 2D landmarks (using MediaPipe) for 3DMM fitting
echo "Step 3: Extracting 2D landmarks for 3DMM fitting..."
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4
echo "2D landmarks extracted and saved as lms_2d.npy"

# Step 4: Fit 3DMM
echo "Step 4: Fitting 3DMM..."
echo "Removed --debug for faster processing: If you want to visualize the fitting process, set --debug flag"
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset --id_mode=global
echo "3DMM fitting done and saved as coeff_fit_mp.npy"

# Step 5: Binarize data
echo "Step 5: Binarizing data..."
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
echo "Data binarized and saved as data/binary/videos/${VIDEO_ID}/trainval_dataset.npy"

# Capture end time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print total time taken in HH:MM:SS format
echo "Data preparation completed for video: ${VIDEO_ID}"
echo "Time taken: $(date -d@${DURATION} -u +%H:%M:%S) (HH:MM:SS)"