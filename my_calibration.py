import cv2, numpy as np



if __name__ == '__main__':
    
    keypoint_inference_path = 'keypoint_inference_result.npz'
    
    keypoint_coords = np.load(keypoint_inference_path)['arr_0']
    
    print(keypoint_coords.shape)