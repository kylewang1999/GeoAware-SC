import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ffmpeg, cv2, torch, json, numpy as np, gc
from os.path import abspath
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from utils.utils_visualization_demo import DemoInteractive, DemoSingleImage
from preprocess_map import set_seed
import torch.nn as nn


set_seed(42)
num_patches = 60
sd_model = sd_aug = extractor_vit = None
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
aggre_net.load_pretrained_weights(torch.load('results_spair/best_856.PTH'))
        
def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def read_mp4(video_path: str) -> np.ndarray:
    """
    Read an MP4 file and return a numpy array of frames.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video
    except Exception as e:
        print(f"Error reading video file {video_path}: {e}")
        return None
    

def imgs_to_mp4(imgs, out_fname='output.mp4', fps=30):
    ''' Convert a batch of images to an mp4 video 
    Inputs:
        imgs: (N,C,H,W) torch.Tensor or np.ndarray, batch of images
    Returns:
        out_fname: str. Path to the output mp4 file
    '''

    assert out_fname.endswith('.mp4'), f"Expect output file to be .mp4, but got {out_fname}"
    assert type(imgs) in [torch.Tensor, np.ndarray], f"Expect input type Torch.tensor or np.ndarray, but got {type(imgs)}"
    assert len(imgs.shape) == 4, f"Expect input shape (N,C,H,W), but got {imgs.shape}"
    assert imgs.shape[1] == 3, f"Expect input shape (N,3,H,W), but got {imgs.shape}"
    
    N, _, H, W = imgs.shape
    imgs = imgs.detach().cpu().numpy() if isinstance(imgs, torch.Tensor) else imgs
    
    if imgs.dtype in [np.float32, np.float64]: 
        assert imgs.min() >= 0 and imgs.max() <= 1, f"Expect input range [0,1], but got {imgs.min()} to {imgs.max()}"
        imgs = (imgs*255).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fname, fourcc, fps, (W, H))
    
    if not out.isOpened():
        print(f"Failed to open video writer with filename {out_fname}")
        return None
    
    for i, img in tqdm(enumerate(imgs), desc="Writing mp4 video", total=len(imgs)):
        out.write(cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Video saved to {abspath(out_fname)}")
    return out_fname

class VideoInference():
    
    def __init__(self, ref_img, ref_kp_coord, mp4_path, img_size, out_video_path='output.mp4', out_keypoint_path='keypoint_inference_result.npz'):
        
        self.ref_img = ref_img
        self.ref_kp_coord = ref_kp_coord
        self.mp4_path = mp4_path    # query each frame in this video
        self.out_video_path = out_video_path
        self.out_keypoint_path = out_keypoint_path

        self.img_size = img_size    # size to resize the video frame
        self.imgs = [resize(Image.fromarray(img), target_res=img_size, resize=True, to_pil=True)\
            for img in read_mp4(mp4_path)]
        
        self.sd_model, self.sd_aug = load_model(diffusion_ver='v1-3', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
        self.extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')

        self.feat_ref = get_processed_features(self.sd_model, self.sd_aug, aggre_net, self.extractor_vit, num_patches, img=self.ref_img)
        
    
    def infer_one_image(self, img):   
        feat_query = get_processed_features(self.sd_model, self.sd_aug, aggre_net, self.extractor_vit, num_patches, img=img)
        num_channel = feat_query.size(1)
        cos = nn.CosineSimilarity(dim=1)
        
        x, y = np.round(self.ref_kp_coord).astype(int)

        with torch.no_grad():

            src_ft = self.feat_ref
            up_sample_scale = self.img_size // src_ft.size(2)
            src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
            src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

            del src_ft
            gc.collect()
            torch.cuda.empty_cache()

            trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(feat_query) # 1, C, H, W
            cos_map = cos(src_vec, trg_ft).cpu().numpy()    # 1, H, W

            del trg_ft
            gc.collect()
            torch.cuda.empty_cache()
   
            max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)

            heatmap = cos_map[0]
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
            
        return max_yx[::-1]
    
    def infer_video(self):
        
        ''' Produces a video with keypoint anotation and a json file with keypoint coordinates '''
        
        annotated_imgs = [] 
        inferred_coords = []
        for img in tqdm(self.imgs):
            inferred_coord_xy = self.infer_one_image(img)
            img = np.array(img)
            img = np.array(cv2.circle(img, inferred_coord_xy, 5, (255, 0, 0), -1))
            annotated_imgs.append(img)
            inferred_coords.append(inferred_coord_xy)
        
        imgs_to_mp4(np.array(annotated_imgs).transpose(0,3,1,2), self.out_video_path)
        np.savez(self.out_keypoint_path, np.stack(inferred_coords))     # (N,2)
        
        
    
    
    


if __name__ == '__main__':
    
    from matplotlib.patches import Circle

    img_size = 240

    # img1_path = 'data/SPair-71k/JPEGImages/dog/2010_000899.jpg' # path to the source image
    img1_path = '/home/kyle/repos/droid_auto_calib/assets/droid_Thu_May_11_13_33_20_2023/img_thumbnails/third_person_1_left.jpg'
    annotation_path = '/home/kyle/repos/droid_auto_calib/assets/droid_Thu_May_11_13_33_20_2023/img_thumbnails/keypoint_annotation.json'
    with open(annotation_path, 'r') as f:
        annotation_dict = json.load(f)
    keypoint_coord = annotation_dict['third_person_1_left.jpg'][0]
    resize_ret = resize(orig_img:=Image.open(img1_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True, return_kp_coord_info=True)
    img1 = resize_ret.canvas
    keypoint_coord[0] = keypoint_coord[0] * resize_ret.x_scale + resize_ret.x_offset
    keypoint_coord[1] = keypoint_coord[1] * resize_ret.y_scale + resize_ret.y_offset
    
    
    
    mp4_path = '/home/kyle/repos/droid_auto_calib/assets/droid_Thu_May_11_13_33_20_2023/recordings/MP4/29838012_left.mp4'
    imgs = read_mp4(mp4_path)
    # img2_path = 'data/SPair-71k/JPEGImages/dog/2011_002398.jpg' # path to the target image
    # img2_path = '/home/kyle/repos/droid_auto_calib/assets/droid_Thu_May_11_13_33_20_2023/img_thumbnails/third_person_1.jpg'
    # img2 = resize(Image.open(img2_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)
    img2 = resize(Image.fromarray(imgs[50]), target_res=img_size, resize=True, to_pil=True)
    
    video_inference = VideoInference(img1, keypoint_coord, mp4_path, img_size)
    
    video_inference.infer_video()

    # # visualize the two images in the same row
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for a in ax: a.axis('off')
    # ax[0].imshow(img1)
    # ax[0].set_title('source image')
    # ax[0].add_patch(Circle(keypoint_coord, radius=10, color='red'))
    # ax[1].imshow(img2)
    # ax[1].set_title('target image')
    # plt.show()
    # plt.savefig('output_srctgt_raw.png')
    
    # sd_model, sd_aug = load_model(diffusion_ver='v1-3', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
    # extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')

    # feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img1)
    # feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img2)
    
    # # demo = Demo([img1,img2], torch.cat([feat1, feat2], dim=0), img_size)
    # # demo.plot_img_pairs(fig_size=5)
    # # demo = DemoSingleImage([img1,img2], torch.cat([feat1, feat2], dim=0), img_size)
    # demo = DemoSingleImage([img1,img2], torch.cat([feat1, feat2], dim=0), img_size)
    # demo.plot_img_pairs(keypoint_coord, fig_size=5)
    
    