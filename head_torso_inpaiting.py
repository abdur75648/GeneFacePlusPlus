# Read head_imgs (green bg) from a directory and torso_imgs (transparent) from another directory and join them to create inpainted images
import os
from tqdm import tqdm
head_imgs_dir = 'Head_Don'
torso_imgs_dir = 'data/processed/videos/Don/inpaint_torso_imgs'
head_images = sorted(os.listdir(head_imgs_dir))
head_images = [os.path.join(head_imgs_dir, img) for img in head_images]
torso_images = sorted(os.listdir(torso_imgs_dir))
torso_images = [os.path.join(torso_imgs_dir, img) for img in torso_images]
len_head = len(head_images)
assert len_head <= len(torso_images)
torso_images = torso_images[:len_head]

import cv2
import numpy as np

import mediapipe as mp
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

    def get_mask(self, img):
        segmap = self.segment_image(img)
        mask = segmap > 0
        return mask

def inpaint_head_torso(head_img, torso_img):
    # Read head
    head = cv2.imread(head_img)
    # Read torso
    torso = cv2.imread(torso_img)
    torso = cv2.imread(torso_img)
    
    # Resize head to match the size of torso
    head = cv2.resize(head, (torso.shape[1], torso.shape[0]))
    assert head.shape == torso.shape == (512, 512, 3)
    
    
    # ### Not accurate method
    # lower_bound = np.array([0, 205, 0])
    # upper_bound = np.array([50, 255, 50])
    # mask = cv2.inRange(head, lower_bound, upper_bound)
    # mask = cv2.bitwise_not(mask)
    # mask = mask // 255
    
    # ### More accurate method
    # lab_head = cv2.cvtColor(head, cv2.COLOR_BGR2LAB)
    # a_channel = lab_head[:,:,1]
    # th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # mask = th // 255
    
    # ### Most accurate method
    seg_model = MediapipeSegmenter()
    mask = seg_model.get_mask(head)
    
    inpainted_image = head*mask[:, :, np.newaxis] + torso*(1-mask)[:, :, np.newaxis]
    
    # Add a green background to the inpainted image
    green_background = np.zeros_like(inpainted_image)
    green_background[..., 1] = 255  # Set green channel to 255
    mask = np.all(inpainted_image == 0, axis=-1)
    inpainted_image = np.where(mask[:, :, None], green_background, inpainted_image).astype(np.uint8)
    ### cv2.imwrite('inpainted_image.jpg', inpainted_image)
    # inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)   
    return inpainted_image

# # Process all images and write as a video 25fps
# import imageio
# tmp_out_name = "headEnhancedResized_InpaintedOrigTorsoImageio.mp4"
# writer = imageio.get_writer(tmp_out_name, fps = 25, format='FFMPEG', codec='h264')

# for i in tqdm(range(len_head)):
#     inpainted_image = inpaint_head_torso(head_images[i], torso_images[i])
#     writer.append_data(inpainted_image)
# writer.close()


# Process all images and write as a video 25fps
tmp_out_name = "headEnhancedResized_InpaintedOrigTorso.mp4"
out = cv2.VideoWriter(tmp_out_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (512, 512))
for idx, head_img in enumerate(tqdm(head_images)):
    torso_img = torso_images[idx]
    inpainted_image = inpaint_head_torso(head_img, torso_img)
    out.write(inpainted_image)
out.release()