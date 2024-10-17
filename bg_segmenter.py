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
        
        base_options = BaseOptions(model_asset_path=model_path, delegate= BaseOptions.Delegate.CPU)
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

        ### Since background is 0, we can directly use the mask as the condition ###
        mask = segmap > 0  # Assuming the category_mask contains 0 for background
        green_screen_img = np.where(mask[:, :, None], img, green_background)
        return green_screen_img

if __name__ == '__main__':
    # # Example usage
    img_path = 'DonImage_1080_1080.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_model = MediapipeSegmenter()
    segmap = seg_model.segment_image(img)
    green_screen_img = seg_model.apply_green_screen(img, segmap)
    cv2.imwrite('DonImage_1080_1080_output_image.jpg', cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
    
    # input_dir = "./x/"
    # output_dir = "./"
    # img_extensions = [".jpg", ".jpeg", ".png"]
    # seg_model = MediapipeSegmenter()
    # ### Without Multiprocessing ###
    # # for img_name in tqdm(os.listdir(input_dir)):
    # #     if not any([img_name.endswith(ext) for ext in img_extensions]):
    # #         continue
    # #     img_path = os.path.join(input_dir, img_name)
    # #     img = cv2.imread(img_path)
    # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # #     segmap = seg_model.segment_image(img)
    # #     green_screen_img = seg_model.apply_green_screen(img, segmap)
    # #     output_path = os.path.join(output_dir, img_name)
    # #     cv2.imwrite(output_path, cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
    # #     print(f"Saved {output_path}")
    
    
    # ### With Multiprocessing ###
    # cpu_count = os.cpu_count()
    # print(f"Using {cpu_count} cores for parallel processing")
    # def process_img(img_name):
    #     if not any([img_name.endswith(ext) for ext in img_extensions]):
    #         return
    #     img_path = os.path.join(input_dir, img_name)
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     segmap = seg_model.segment_image(img)
    #     green_screen_img = seg_model.apply_green_screen(img, segmap)
    #     output_path = os.path.join(output_dir, img_name)
    #     cv2.imwrite(output_path, cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR))
    #     print(f"Saved {output_path}")
    
    # with Pool(cpu_count) as p:
    #     list(tqdm(p.imap(process_img, os.listdir(input_dir)), total=len(os.listdir(input_dir))) )