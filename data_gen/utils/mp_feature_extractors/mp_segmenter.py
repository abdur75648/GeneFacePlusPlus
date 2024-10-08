import os
import cv2
import numpy as np
import tqdm
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from utils.commons.multiprocess_utils import multiprocess_run_tqdm, multiprocess_run
from utils.commons.tensor_utils import convert_to_np
from sklearn.neighbors import NearestNeighbors

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

def scatter_np(condition_img, classSeg=5):
# def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.shape
    # if height != label_size[0] or width != label_size[1]:
        # condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = np.zeros([batch, classSeg, condition_img.shape[2], condition_img.shape[3]]).astype(np.int_)
    # input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    np.put_along_axis(input_label, condition_img, 1, 1)
    return input_label

def scatter(condition_img, classSeg=19):
# def scatter(condition_img, classSeg=19, label_size=(512, 512)):
    batch, c, height, width = condition_img.size()
    # if height != label_size[0] or width != label_size[1]:
        # condition_img= F.interpolate(condition_img, size=label_size, mode='nearest')
    input_label = torch.zeros(batch, classSeg, condition_img.shape[2], condition_img.shape[3], device=condition_img.device)
    # input_label = torch.zeros(batch, classSeg, *label_size, device=condition_img.device)
    return input_label.scatter_(1, condition_img.long(), 1)

def encode_segmap_mask_to_image(segmap):
    # rgb
    _,h,w = segmap.shape
    encoded_img = np.ones([h,w,3],dtype=np.uint8) * 255
    colors = [(255,255,255),(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0)]
    for i, color in enumerate(colors):
        mask = segmap[i].astype(int)
        index = np.where(mask != 0)
        encoded_img[index[0], index[1], :] = np.array(color)
    return encoded_img.astype(np.uint8)
        
def decode_segmap_mask_from_image(encoded_img):
    # rgb
    colors = [(255,255,255),(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0)]
    bg = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 255)
    hair = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 0)
    body_skin = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 0) & (encoded_img[..., 2] == 255)
    face_skin = (encoded_img[..., 0] == 0) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 255)
    clothes = (encoded_img[..., 0] == 255) & (encoded_img[..., 1] == 0) & (encoded_img[..., 2] == 0)
    others = (encoded_img[..., 0] == 0) & (encoded_img[..., 1] == 255) & (encoded_img[..., 2] == 0)
    segmap = np.stack([bg, hair, body_skin, face_skin, clothes, others], axis=0)
    return segmap.astype(np.uint8)

def read_video_frame(video_name, frame_id):
    # https://blog.csdn.net/bby1987/article/details/108923361
    # frame_num = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) # ==> 总帧数
    # fps = video_capture.get(cv2.CAP_PROP_FPS)               # ==> 帧率
    # width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)     # ==> 视频宽度
    # height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   # ==> 视频高度
    # pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)        # ==> 句柄位置
    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1000)        # ==> 设置句柄位置
    # pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)        # ==> 此时 pos = 1000.0
    # video_capture.release()
    vr = cv2.VideoCapture(video_name)
    vr.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = vr.read()
    return frame

def decode_segmap_mask_from_segmap_video_frame(video_frame):
    # video_frame: 0~255 BGR, obtained by read_video_frame
    def assign_values(array):
        remainder = array % 40  # 计算数组中每个值与40的余数
        assigned_values = np.where(remainder <= 20, array - remainder, array + (40 - remainder))
        return assigned_values
    segmap = video_frame.mean(-1)
    segmap = assign_values(segmap) // 40 # [H, W] with value 0~5 
    segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
    return segmap.astype(np.uint8)

### Usage fo segmenter model
# seg_model = MediapipeSegmenter2()
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# segmap = seg_model.segment_image(img)

segmenter_helper = MediapipeSegmenter()
def job_cal_seg_map_for_image(img, segmenter_options=None, segmenter=None):
    """
    被 MediapipeSegmenter.multiprocess_cal_seg_map_for_a_video所使用, 专门用来处理单个长视频.
    """
    # segmenter_actual = segmenter_helper
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    # out = segmenter_actual.segment(mp_image)
    # segmap = out.category_mask.numpy_view().copy() # [H, W]
    
    segmap = segmenter_helper.segment_image(img)


    ### print("segmap: ", segmap.shape) # (512, 512)
    ### print("segmap unique: ", np.unique(segmap)) # segmap unique:  [0 1 2 3 4 5] [bg, hair, body, face, clothes, others]
    ### import cv2
    ### segmap_img1 = (segmap*40).astype(np.uint8)
    ### cv2.imwrite("segmap1.png", segmap_img1)
    ### ### Find a mask with all pixels with value 3, find it's upper 2/3rd and correct the body pixels
    face_mask = segmap == 3
    y, x = np.where(face_mask)
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    y_two_third = y_min + 2*(y_max - y_min) // 3
    body_mask = segmap == 2
    body_mask[y_two_third:, :] = 0
    segmap[body_mask] = 3
    ### segmap_img2 = (segmap*40).astype(np.uint8)
    ### cv2.imwrite("segmap2.png", segmap_img2)

    segmap_mask = scatter_np(segmap[None, None, ...], classSeg=6)[0] # [6, H, W]
    segmap_image = segmap[:, :, None].repeat(3, 2).astype(float)
    segmap_image = (segmap_image * 40).astype(np.uint8)

    return segmap_mask, segmap_image