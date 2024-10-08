import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
import copy
import glob
import cv2
import tqdm
import numpy as np
from typing import Union
from utils.commons.tensor_utils import convert_to_np
from utils.commons.os_utils import multiprocess_glob
import pickle
import traceback
import multiprocessing
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from data_gen.utils.mp_feature_extractors.mp_segmenter import encode_segmap_mask_to_image
from data_gen.utils.mp_feature_extractors.mp_segmenter import decode_segmap_mask_from_image
from data_gen.utils.mp_feature_extractors.mp_segmenter import job_cal_seg_map_for_image
from data_gen.utils.process_video.split_video_to_imgs import extract_img_job

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 
        
def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content

def save_rgb_alpha_image_to_path(img, alpha, img_path):
    try: os.makedirs(os.path.dirname(img_path), exist_ok=True)
    except: pass
    cv2.imwrite(img_path, np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), alpha], axis=-1))

def save_rgb_image_to_path(img, img_path):
    try: os.makedirs(os.path.dirname(img_path), exist_ok=True)
    except: pass
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_rgb_image_to_path(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def image_similarity(x: np.ndarray, y: np.ndarray, method="mse"):
    if method == "mse":
        return np.mean((x - y) ** 2)
    else:
        raise NotImplementedError

def inpaint_torso_job(gt_img, segmap):
    bg_part = (segmap[0]).astype(bool)
    head_part = (segmap[1] + segmap[3] + segmap[5]).astype(bool)
    neck_part = (segmap[2]).astype(bool)
    torso_part = (segmap[4]).astype(bool) 
    img = gt_img.copy()
    img[head_part] = 0
    
    ## Create a random prefix for all the image names
    # img_name_prefix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=5))
    
    ## Save the original torso image
    # img_to_be_saved = (img * 255).astype(np.uint8)
    # out_img_name = f"{img_name_prefix}_torso_orig.jpg"
    # cv2.imwrite(out_img_name, img_to_be_saved)

    # torso part "vertical" in-painting...
    L = 8 + 1
    torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
    # lexsort: sort 2D coords first by y then by x, 
    # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
    torso_coords = torso_coords[inds]
    # choose the top pixel for each column
    u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
    top_torso_coords = torso_coords[uid] # [m, 2]
    # only keep top-is-head pixels
    top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0]) # [N, 2]
    mask = head_part[tuple(top_torso_coords_up.T)] 
    if mask.any():
        top_torso_coords = top_torso_coords[mask]
        # get the color
        top_torso_colors = gt_img[tuple(top_torso_coords.T)] # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_torso_coords += inpaint_offsets
        inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
        inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
        inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
        # set color
        img[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors
        inpaint_torso_mask = np.zeros_like(img[..., 0]).astype(bool)
        inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
    else:
        inpaint_torso_mask = None
    
    # torso_img_intermediate = img.copy()
    # torso_img_intermediate[inpaint_torso_mask] = cv2.GaussianBlur(torso_img_intermediate, (5, 5), cv2.BORDER_DEFAULT)[inpaint_torso_mask]
    # torso_img_intermediate[~inpaint_torso_mask] = 0
    # img_name_prefix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=5))
    # out_img_name = f"{img_name_prefix}_torso_intermediate.jpg"
    # torso_img_intermediate = (torso_img_intermediate * 255).astype(np.uint8)
    # cv2.imwrite(out_img_name, cv2.cvtColor(torso_img_intermediate[0], cv2.COLOR_RGB2BGR))
    # img_to_be_saved = (img * 255).astype(np.uint8)
    # out_img_name = f"{img_name_prefix}_torso_2.jpg"
    # cv2.imwrite(out_img_name, img_to_be_saved)
    
    # neck part "vertical" in-painting...
    push_down = 4
    L = 64 + push_down + 1
    neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)
    neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
    # lexsort: sort 2D coords first by y then by x, 
    # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
    neck_coords = neck_coords[inds]
    # choose the top pixel for each column
    u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
    top_neck_coords = neck_coords[uid] # [m, 2]
    # only keep top-is-head pixels
    top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
    mask = head_part[tuple(top_neck_coords_up.T)] 
    top_neck_coords = top_neck_coords[mask]
    # push these top down for 4 pixels to make the neck inpainting more natural...
    offset_down = np.minimum(ucnt[mask] - 1, push_down)
    top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
    # get the color
    top_neck_colors = gt_img[tuple(top_neck_coords.T)] # [m, 3]
    # construct inpaint coords (vertically up, or minus in x)
    inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
    inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
    inpaint_neck_coords += inpaint_offsets
    inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]
    inpaint_neck_colors = top_neck_colors[None].repeat(L, 0) # [L, m, 3]
    darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
    inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
    # set color
    img[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors
    # apply blurring to the inpaint area to avoid vertical-line artifects...
    inpaint_mask = np.zeros_like(img[..., 0]).astype(bool)
    inpaint_mask[tuple(inpaint_neck_coords.T)] = True
    
    # img_to_be_saved = (img * 255).astype(np.uint8)
    # out_img_name = f"{img_name_prefix}_torso_3.jpg"
    # cv2.imwrite(out_img_name, img_to_be_saved)

    blur_img = img.copy()
    blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)
    img[inpaint_mask] = blur_img[inpaint_mask]

    # set mask
    torso_img_mask = (neck_part | torso_part | inpaint_mask)
    torso_with_bg_img_mask = (bg_part | neck_part | torso_part | inpaint_mask)
    if inpaint_torso_mask is not None:
        torso_img_mask = torso_img_mask | inpaint_torso_mask
        torso_with_bg_img_mask = torso_with_bg_img_mask | inpaint_torso_mask
    
    torso_img = img.copy()
    torso_img[~torso_img_mask] = 0
    torso_with_bg_img = img.copy()
    torso_img[~torso_with_bg_img_mask] = 0

    return torso_img, torso_img_mask, torso_with_bg_img, torso_with_bg_img_mask

def load_segment_mask_from_file(filename: str):
    encoded_segmap = load_rgb_image_to_path(filename)
    segmap_mask = decode_segmap_mask_from_image(encoded_segmap)
    return segmap_mask

# load segment mask to memory if not loaded yet
def refresh_segment_mask(segmap_mask: Union[str, np.ndarray]):
    if isinstance(segmap_mask, str):
        segmap_mask = load_segment_mask_from_file(segmap_mask)
    return segmap_mask

# load segment mask to memory if not loaded yet
def refresh_image(image: Union[str, np.ndarray]):
    if isinstance(image, str):
        image = load_rgb_image_to_path(image)
    return image

def seg_out_img_with_segmap(img, segmap, mode='head'):
    """
    img: [h,w,c], img is in 0~255, np
    """
    # 
    img = copy.deepcopy(img)
    if mode == 'head':
        selected_mask = segmap[[1,3,5] , :, :].sum(axis=0)[None,:] > 0.5 # glasses 也属于others
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
        # selected_mask = segmap[[1,3] , :, :].sum(dim=0, keepdim=True) > 0.5
    elif mode == 'person':
        selected_mask = segmap[[1,2,3,4,5], :, :].sum(axis=0)[None,:] > 0.5 
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
    elif mode == 'torso':
        selected_mask = segmap[[2,4], :, :].sum(axis=0)[None,:] > 0.5
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
    elif mode == 'torso_with_bg':
        selected_mask = segmap[[0, 2,4], :, :].sum(axis=0)[None,:] > 0.5
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
    elif mode == 'bg':
        selected_mask = segmap[[0], :, :].sum(axis=0)[None,:] > 0.5  # only seg out 0, which means background
        img[~selected_mask.repeat(3,axis=0).transpose(1,2,0)] = 0 # (-1,-1,-1) denotes black in our [-1,1] convention
    elif mode == 'full':
        pass
    else:
        raise NotImplementedError()
    return img, selected_mask

def generate_segment_imgs_job(img_name, segmap, img):
    out_img_name = segmap_name = img_name.replace("/gt_imgs/", "/segmaps/").replace(".jpg", ".png") # 存成jpg的话，pixel value会有误差
    try: os.makedirs(os.path.dirname(out_img_name), exist_ok=True)
    except: pass
    encoded_segmap = encode_segmap_mask_to_image(segmap)
    save_rgb_image_to_path(encoded_segmap, out_img_name)

    for mode in ['head', 'torso', 'person', 'bg']:
        out_img, mask = seg_out_img_with_segmap(img, segmap, mode=mode)
        img_alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) # alpha
        mask = mask[0][..., None]
        img_alpha[~mask] = 0
        out_img_name = img_name.replace("/gt_imgs/", f"/{mode}_imgs/").replace(".jpg", ".png") # 1024 resolution
        save_rgb_alpha_image_to_path(out_img, img_alpha, out_img_name)
        out_img = cv2.resize(out_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_alpha = cv2.resize(img_alpha, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_alpha = img_alpha[..., None]
        out_img_name = img_name.replace("_1024/gt_imgs/", f"_512/{mode}_imgs/").replace(".jpg", ".png") # 512 resolution
        save_rgb_alpha_image_to_path(out_img, img_alpha, out_img_name)
    
    inpaint_torso_img, inpaint_torso_img_mask, inpaint_torso_with_bg_img, inpaint_torso_with_bg_img_mask = inpaint_torso_job(img, segmap)
    img_alpha = 255 * np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) # alpha
    img_alpha[~inpaint_torso_img_mask[..., None]] = 0
    out_img_name = img_name.replace("/gt_imgs/", f"/inpaint_torso_imgs/").replace(".jpg", ".png") # 1024 resolution
    save_rgb_alpha_image_to_path(inpaint_torso_img, img_alpha, out_img_name)
    out_img = cv2.resize(inpaint_torso_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_alpha = cv2.resize(img_alpha, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_alpha = img_alpha[..., None]
    out_img_name = img_name.replace("_1024/gt_imgs/", f"_512/inpaint_torso_imgs/").replace(".jpg", ".png") # 512 resolution
    save_rgb_alpha_image_to_path(out_img, img_alpha, out_img_name)
    return segmap_name
    
def segment_and_generate_for_image_job(img_name, img, segmenter_options=None, segmenter=None, store_in_memory=False):
    img = refresh_image(img)
    segmap_mask, segmap_image = job_cal_seg_map_for_image(img, segmenter_options=segmenter_options, segmenter=segmenter)
    segmap_name = generate_segment_imgs_job(img_name=img_name, segmap=segmap_mask, img=img)
    if store_in_memory:
        return segmap_mask
    else:
        return segmap_name

def segment_and_generate_for_image_job_helper(arg):
        img_name, img = arg
        return segment_and_generate_for_image_job(img_name, img)#, None, None, False) -> Default
    
def extract_segment_job(
    video_name, 
    nerf=False, 
    background_method='knn', 
    device="cpu",
    total_gpus=0, 
    mix_bg=True,
    store_in_memory=False, # set to True to speed up a bit of preprocess, but leads to HUGE memory costs (100GB for 5-min video)
    force_single_process=False, # turn this on if you find multi-process does not work on your environment
    num_processes=16
):
    
    multiprocess_enable = False
    
    if "cuda" in device:
        raise NotImplementedError("CUDA is not supported in this version")
        # determine which cuda index from subprocess id
        pname = multiprocessing.current_process().name
        pid = int(pname.rsplit("-", 1)[-1]) - 1
        cuda_id = pid % total_gpus
        device = f"cuda:{cuda_id}"

    if nerf: # single video
        raw_img_dir = video_name.replace(".mp4", "_1024/gt_imgs/").replace("/raw/","/processed/")
    else:
        raise NotImplementedError()
    if not os.path.exists(raw_img_dir):
        extract_img_job(video_name, raw_img_dir) # use ffmpeg to split video into imgs
    
    img_names = glob.glob(os.path.join(raw_img_dir, "*.jpg"))

    img_lst = []

    for img_name in img_names:
        if store_in_memory:
            img = load_rgb_image_to_path(img_name)
        else:
            img = img_name
        img_lst.append(img)

    print("| Extracting Segmaps && Saving...")
    args = []
    segmap_mask_lst = []
    # preparing parameters for segment
    for i in range(len(img_lst)):
        img_name = img_names[i]
        img = img_lst[i]
        options = None
        segmenter_arg = None
        arg = (img_name, img)
        args.append(arg)
        
    # if multiprocess_enable:
    #     print("="*20)
    #     print("Multiprocess enabled with num_workers = 2")
    #     print("="*20)
    #     for (_, res) in multiprocess_run_tqdm(segment_and_generate_for_image_job, args=args, num_workers=2, desc='generating segment images in multi-processes...'):
    #         segmap_mask = res
    #         segmap_mask_lst.append(segmap_mask)
    # else:
    #     print("="*20)
    #     print("Single Process segment_and_generate_for_image_job!")
    #     print("="*20)
    #     for index in tqdm.tqdm(range(len(img_lst)), desc="generating segment images in single-process..."):
    #         segmap_mask = segment_and_generate_for_image_job(*args[index])
    #         segmap_mask_lst.append(segmap_mask)
    # print("| Extracted Segmaps Done.")
    

    print("Multiprocess enabled with num_workers = ", num_processes)
    print("="*20)
    with multiprocessing.Pool(num_processes) as pool:
        for segmap_mask in tqdm.tqdm(pool.imap(segment_and_generate_for_image_job_helper, args), total=len(args), desc="Generating segment images..."):
            segmap_mask_lst.append(segmap_mask)
    print("="*20)

    
    print("| Extracting background... [Harcoded to green bg]!!!")
    bg_prefix_name = "bg"
    # bg_img = extract_background(img_lst, segmap_mask_lst, method=background_method, device=device, mix_bg=mix_bg)
    bg_img = np.zeros_like(refresh_image(img_lst[0]))
    bg_img[..., 1] = 255
    if nerf:
        out_img_name = video_name.replace("/raw/", "/processed/").replace(".mp4", f"_1024/{bg_prefix_name}.jpg")
    else:
        raise NotImplementedError()
    save_rgb_image_to_path(bg_img, out_img_name) # 1024 resolution
    bg_img_resized = cv2.resize(bg_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    out_img_name = out_img_name.replace("_1024/", "_512/")
    save_rgb_image_to_path(bg_img_resized, out_img_name) # 512 resolution
    print("| Extracted background done.")
    
    print("| Extracting com_imgs...")
    com_prefix_name = f"com"
    print("="*20)
    print("Multiprocess enabled with num_workers = ", num_processes)
    print("="*20)
    com_args = [(img_names[i], img_lst[i], segmap_mask_lst[i], bg_img, com_prefix_name) for i in range(len(img_names))]
    
    def generate_com_img(arg):
        img_name, img, segmap, bg_img, com_prefix_name = arg
        com_img = refresh_image(img).copy()
        segmap = refresh_segment_mask(segmap)
        bg_part = segmap[0].astype(bool)[..., None].repeat(3, axis=-1)
        com_img[bg_part] = bg_img[bg_part]
        out_img_name = img_name.replace("/gt_imgs/", f"/{com_prefix_name}_imgs/")
        save_rgb_image_to_path(com_img, out_img_name)  # 1024 resolution
        com_img = cv2.resize(com_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        out_img_name = img_name.replace("_1024/gt_imgs/", f"_512/{com_prefix_name}_imgs/")
        save_rgb_image_to_path(com_img, out_img_name)  # 512 resolution
        return out_img_name

    with multiprocessing.Pool(num_processes) as pool:
        for _ in tqdm.tqdm(pool.imap(generate_com_img, com_args), total=len(com_args), desc="Extracting com_imgs..."):
            pass

    print("| Extracted com_imgs done.")
    
    return 0

def out_exist_job(vid_name, background_method='knn'):
    com_prefix_name = f"com"
    img_dir = vid_name.replace("/video/", "/gt_imgs/").replace(".mp4", "")
    out_dir1 = img_dir.replace("/gt_imgs/", "/head_imgs/")
    out_dir2 = img_dir.replace("/gt_imgs/", f"/{com_prefix_name}_imgs/")
    
    if os.path.exists(img_dir) and os.path.exists(out_dir1) and os.path.exists(out_dir1) and os.path.exists(out_dir2) :
        num_frames = len(os.listdir(img_dir))
        if len(os.listdir(out_dir1)) == num_frames and len(os.listdir(out_dir2)) == num_frames:
            return None
        else:
            return vid_name
    else:
        return vid_name

def get_todo_vid_names(vid_names, background_method='knn'):
    if len(vid_names) == 1: # nerf
        return vid_names
    todo_vid_names = []
    fn_args = [(vid_name, background_method) for vid_name in vid_names]
    for i, res in multiprocess_run_tqdm(out_exist_job, fn_args, num_workers=4, desc="checking todo videos..."):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names

if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", default='data/raw/videos/May.mp4')
    parser.add_argument("--ds_name", default='nerf')
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--load_names", action="store_true")
    parser.add_argument("--background_method", choices=['knn', 'mat', 'ddnm', 'lama'], type=str, default='knn')
    parser.add_argument("--total_gpus", default=0, type=int) # zero gpus means utilizing cpu
    parser.add_argument("--no_mix_bg", action="store_true")
    parser.add_argument("--store_in_memory", action="store_true") # set to True to speed up preprocess, but leads to high memory costs
    parser.add_argument("--force_single_process", action="store_true") # turn this on if you find multi-process does not work on your environment

    args = parser.parse_args()
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    load_names = args.load_names
    background_method = args.background_method
    total_gpus = args.total_gpus
    mix_bg = not args.no_mix_bg
    store_in_memory = args.store_in_memory
    force_single_process = args.force_single_process

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    for d in devices[:total_gpus]:
        os.system(f'pkill -f "voidgpu{d}"')
        
    if ds_name.lower() == 'nerf': # 处理单个视频
        vid_names = [vid_dir]
        out_names = [video_name.replace("/raw/", "/processed/").replace(".mp4","_lms.npy") for video_name in vid_names]
    else: # 处理整个数据集
        if ds_name in ['lrs3_trainval']:
            vid_name_pattern = os.path.join(vid_dir, "*/*.mp4")
        elif ds_name in ['TH1KH_512', 'CelebV-HQ']:
            vid_name_pattern = os.path.join(vid_dir, "*.mp4")
        elif ds_name in ['lrs2', 'lrs3', 'voxceleb2']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        elif ds_name in ["RAVDESS", 'VFHQ']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        else:
            raise NotImplementedError()
        
        vid_names_path = os.path.join(vid_dir, "vid_names.pkl")
        if os.path.exists(vid_names_path) and load_names:
            print(f"loading vid names from {vid_names_path}")
            vid_names = load_file(vid_names_path)
        else:
            vid_names = multiprocess_glob(vid_name_pattern)
        vid_names = sorted(vid_names)
        print(f"saving vid names to {vid_names_path}")
        save_file(vid_names_path, vid_names)

    vid_names = sorted(vid_names)
    random.seed(args.seed)
    random.shuffle(vid_names)

    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    if not args.reset:
        vid_names = get_todo_vid_names(vid_names, background_method)
    print(f"todo videos number: {len(vid_names)}")

    ### Simple Code
    # device = "cuda" if total_gpus > 0 else "cpu"
    # extract_job = extract_segment_job
    # fn_args = [(vid_name, ds_name=='nerf', background_method, device, total_gpus, mix_bg, store_in_memory, force_single_process) for i, vid_name in enumerate(vid_names)]
        
    # if ds_name == 'nerf': # 处理单个视频
    #     extract_job(*fn_args[0])
    # else:
    #     for vid_name in multiprocess_run_tqdm(extract_job, fn_args, desc=f"Root process {args.process_id}:  segment images", num_workers=args.num_workers):
    #         pass
    
    device = "cpu"
    num_workers = min(args.num_workers, max(os.cpu_count()-2, 1))
    extract_segment_job(vid_names[0], ds_name=='nerf', background_method, device, total_gpus, mix_bg, store_in_memory, force_single_process, num_workers)
    print(f"Process of extracting segment images done for {vid_names[0]}")