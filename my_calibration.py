
import os, sys, cv2, numpy as np, functools, yaml, json, warnings, open3d as o3d
import pybullet as pb
import pybullet_data
from os.path import dirname, abspath, join, exists
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from types import SimpleNamespace as NSpace
from utils.utils_calibration_solver import (
    solve_pnp, solve_pnp_ransac, rtvec_to_matrix, apply_se3_mat, apply_intrinsics_mat
)


sys.path.append(abspath('./third_party/droid'))
from third_party.droid.droid.trajectory_utils.misc import load_trajectory
from third_party.droid.droid.data_processing.timestep_processing import TimestepProcesser
from third_party.droid.droid.data_loading.tf_data_loader import get_type_spec, get_tf_dataloader


def dict2namespace(d):
    ''' Convert dictionary to namespace '''
    if isinstance(d, dict):
        return NSpace(**{k: dict2namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict2namespace(v) for v in d]
    else:
        return d

@functools.lru_cache(maxsize=None)
def load_urdf_in_pybullet(urdf_file, echo=False):
    ''' Reference: https://github.com/robotflow-initiative/Kalib.git '''
    physicsClient = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )
    if echo:
        print(f"The loaded urdf file is {urdf_file}")

    # with suppress_stdout():
    robotId = pb.loadURDF(urdf_file)  # Load the URDF file
    return robotId


def compute_forward_kinematics(
    urdf_file, joint_positions, link_indices=None, return_pose=False, echo=False
):
    ''' Reference: https://github.com/robotflow-initiative/Kalib.git '''
    robotId = load_urdf_in_pybullet(urdf_file, echo)
    # Set joint positions
    for joint_index, position in enumerate(joint_positions):
        pb.resetJointState(robotId, jointIndex=joint_index, targetValue=position)

    # Compute forward kinematics
    link_positions = []
    if return_pose:
        link_mats = []

    if echo:
        print(f"the number of joints in panda.urdf is {pb.getNumJoints(robotId)}")
        print(f"the number of links in panda.urdf is {pb.getNumJoints(robotId)}")

    for joint_index in range(pb.getNumJoints(robotId)):
        link_state = pb.getLinkState(
            robotId, joint_index, computeForwardKinematics=True
        )
        link_position, link_quat = list(link_state[0]), list(link_state[1])
        if return_pose:
            rot_mat = pb.getMatrixFromQuaternion(link_quat)
            rot_mat = np.array(rot_mat).reshape(3, 3)
            trans_mat = np.eye(4)
            trans_mat[:3, 3] = link_position
            trans_mat[:3, :3] = rot_mat
            link_mats.append(trans_mat)
        link_positions.append(np.array(link_position))

    link_positions = np.stack(link_positions)
    if return_pose:
        link_mats = np.stack(link_mats)
        return link_positions, link_mats
    if link_indices is not None:
        link_positions = link_positions[link_indices]
    return link_positions


def convert_raw_extrinsics_to_mat(raw_data):
    # Preprocessing of the extrinsics
    pos = raw_data[0:3]
    rot_mat = R.from_euler("xyz", raw_data[3:6]).as_matrix()

    raw_data = np.zeros((4, 4))
    raw_data[:3, :3] = rot_mat
    raw_data[:3, 3] = pos
    raw_data[3, 3] = 1.0
    raw_data = np.linalg.inv(raw_data)
    return raw_data


class TrajectoryDataset():
    
    def __init__(self, 
                 data_folder_path="./assets/droid_examples/droid_Thu_May_11_13_33_20_2023",
                 recording_prefix="SVO",
                 traj_loading_kwargs={},
                 timestep_filtering_kwargs={},
                 image_transform_kwargs={},
                 camera_kwargs={}):
        
        h5_path = join(data_folder_path, 'trajectory.h5')
        recordings_dir = join(data_folder_path, 'recordings', recording_prefix)
        timesteps = load_trajectory(h5_path, 
                                    recording_folderpath=recordings_dir, 
                                    camera_kwargs=camera_kwargs, 
                                    **traj_loading_kwargs)
        
        self.data_folder_path = data_folder_path
        self.recording_prefix = recording_prefix
        self.timestep_processer = TimestepProcesser(image_transform_kwargs=image_transform_kwargs,
                                                    **timestep_filtering_kwargs)
        self.timesteps =  [self.timestep_processer.forward(t, concat_states=False) for t in timesteps]
        
    def __len__(self): return len(self.timesteps)
        

class CalibratorCam2Base():
    
    def __init__(self, trajectory_dataset, subsample_stride=1):

        self.trajectory_dataset = trajectory_dataset
        self.data_dir = trajectory_dataset.data_folder_path
        self.timesteps = trajectory_dataset.timesteps
        self.cam_kp_coords = None

        cam_kp_coords_file = join(self.data_dir, 'keypoint_inference_result.npz')
        if exists(cam_kp_coords_file):
            self.cam_kp_coords = np.load(cam_kp_coords_file)['arr_0']
            assert len(self.cam_kp_coords) == len(self.timesteps), f"Keypoint inference result length {len(self.cam_kp_coords)} mismatches with number of timesteps {len(self.timesteps)}"
        else:
            warnings.warn(f"Keypoint inference result not found at {cam_kp_coords_file}. Need to be inferred/tracked")
            
        self.cam_kp_coords = self.cam_kp_coords[::subsample_stride]
        self.timesteps = self.timesteps[::subsample_stride]
        
        cherry_pick_timestep_inds = [3,4,5,6]
        self.cam_kp_coords = [self.cam_kp_coords[i] for i in cherry_pick_timestep_inds]
        self.timesteps = [self.timesteps[i] for i in cherry_pick_timestep_inds]

        # Configs
        with open('./configs/calibration_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.config = dict2namespace(config)

        with open(self.config.kalib_config_path, 'r') as file:
            config_kalib = json.load(file)
        self.config_kalib = config_kalib    # a template json file

        # Save paths
        self.save_paths = {key: join(self.config.output_dir, key) for key in self.config.configured_cams}
        self.save_imgs = {key: [] for key in self.config.configured_cams}
        self.debug_save_imgs = {key: [] for key in self.config.configured_cams}

        if not exists(self.config.output_dir):
            for val in self.save_paths.values(): 
                os.makedirs(val, exist_ok=True)
                os.makedirs(join(val, 'debug'), exist_ok=True)

    
    def clean_up(self): os.rmdir(self.config.output_dir)
        
        
    def kalibify_timestep(self, timestep):
        ''' Convert droid timestep to kalib format '''
        
        extrinsics_dict = timestep["extrinsics_dict"]
        intrinsics_dict = timestep["intrinsics_dict"]
        # timestep["observation"] contains keys: 'cartesian_position', 'gripper_position', 'joint_positions', 'joint_torques_computed', 'joint_velocities', 'motor_torques_measured', 'prev_command_successful', 'prev_controller_latency_ms', 'prev_joint_torques_computed', 'prev_joint_torques_computed_safened'

        obs = {
            "robot_state/cartesian_position": timestep["observation"]["robot_state"]["cartesian_position"][:3],
            "robot_state/joint_positions": timestep["observation"]["robot_state"]["joint_positions"],
            "robot_state/gripper_position": [timestep["observation"]["robot_state"]["gripper_position"]],  # wrap as array, raw data is single float
            "camera/img/varied_camera_1_left_img": timestep["observation"]["camera"]["image"]["varied_camera"][0],
            "camera/img/varied_camera_1_right_img": timestep["observation"]["camera"]["image"]["varied_camera"][1],
            "camera/img/varied_camera_2_left_img": timestep["observation"]["camera"]["image"]["varied_camera"][2],
            "camera/img/varied_camera_2_right_img": timestep["observation"]["camera"]["image"]["varied_camera"][3],
            "camera/extrinsics/varied_camera_1_left": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][0]),
            "camera/extrinsics/varied_camera_1_right": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][1]),
            "camera/extrinsics/varied_camera_2_left": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][2]),
            "camera/extrinsics/varied_camera_2_right": convert_raw_extrinsics_to_mat(extrinsics_dict["varied_camera"][3]),
            "camera/intrinsics/varied_camera_1_left": intrinsics_dict["varied_camera"][0],
            "camera/intrinsics/varied_camera_1_right": intrinsics_dict["varied_camera"][1],
            "camera/intrinsics/varied_camera_2_left": intrinsics_dict["varied_camera"][2],
            "camera/intrinsics/varied_camera_2_right": intrinsics_dict["varied_camera"][3]
        }

        for k in obs: obs[k] = np.array(obs[k])

        return obs

    
    def prepare_kalib_data(self):
    
        for timestep_i, timestep in (pbar:=tqdm(enumerate(self.timesteps), total=len(self.timesteps))):
            
            pbar.update(1)
            timestep = self.kalibify_timestep(timestep)

            joint_positions = timestep["robot_state/joint_positions"]
            raw_eef_pos = timestep["robot_state/cartesian_position"]            
            eefpos_from_joint_positions = compute_forward_kinematics(urdf_file=self.config.urdf_path, 
                                                                     joint_positions=joint_positions)   # (num_joints, 3)
            eefpos_from_joint_positions = eefpos_from_joint_positions[-1] # (3,) gripper's position
            eefpos = np.array(eefpos_from_joint_positions)
            robot_joints = eefpos[None]

            # available options: 'varied_camera_1_left', 'varied_camera_1_right', 'varied_camera_2_left', 'varied_camera_2_right'
            for camera in self.config.configured_cams:
                camera_intrinsics = timestep["camera/intrinsics/" + camera]
                camera_extrinsics = timestep["camera/extrinsics/" + camera]

                projected_eefpos = np.hstack([eefpos, 1])
                projected_eefpos = np.dot(camera_extrinsics, projected_eefpos)
                projected_eefpos = np.dot(camera_intrinsics, projected_eefpos[:3])
                projected_eefpos = projected_eefpos[:2] / projected_eefpos[2]

                projected_raw_eefpos = np.hstack([raw_eef_pos, 1])
                projected_raw_eefpos = np.dot(camera_extrinsics, projected_raw_eefpos)
                projected_raw_eefpos = np.dot(camera_intrinsics, projected_raw_eefpos[:3])
                projected_raw_eefpos = projected_raw_eefpos[:2] / projected_raw_eefpos[2]

                self.save_imgs[camera] = timestep["camera/img/" + camera + "_img"]
                debug_img = self.save_imgs[camera][...,:3].copy()
                # debug_img = cv2.circle(debug_img, tuple(projected_eefpos.astype(int)), 5, (0, 0, 255), -1)
                # debug_img = cv2.circle(debug_img, tuple(projected_raw_eefpos.astype(int)), 5, (0, 255, 255), -1)
                debug_img = cv2.circle(debug_img, tuple(self.cam_kp_coords[timestep_i].astype(int)), 10, (0, 255, 0), -1) # TODO: Separate kp_coords according to camera key

                self.debug_save_imgs[camera] = debug_img

                cv2.imwrite(join(self.save_paths[camera], f"{timestep_i:06d}_{camera}.png"), self.save_imgs[camera])
                cv2.imwrite(join(self.save_paths[camera], f"debug/{timestep_i:06d}_{camera}.png"), self.debug_save_imgs[camera])

                self.config_kalib["objects"][0]["eef_pos"] = eefpos.tolist()
                self.config_kalib["objects"][0]["camera_intrinsics"] = camera_intrinsics.tolist()
                self.config_kalib["objects"][0]["local_to_world_matrix"] = camera_extrinsics.tolist()
                self.config_kalib["objects"][0]["joint_positions"] = timestep[
                    "robot_state/joint_positions"
                ].tolist()
                self.config_kalib["objects"][0]["cartesian_position"] = timestep[
                    "robot_state/cartesian_position"
                ].tolist()

                for ptidx, pt in enumerate(robot_joints):
                    self.config_kalib["objects"][0]["keypoints"][ptidx]["location"] = eefpos.tolist()
                    self.config_kalib["objects"][0]["keypoints"][ptidx]["name"] = "panda_left_finger"
                    self.config_kalib["objects"][0]["keypoints"][ptidx]["projected_location"] = projected_eefpos.tolist()
                    if self.cam_kp_coords is not None:
                        self.config_kalib["objects"][0]["keypoints"][ptidx]["predicted_location"] = self.cam_kp_coords[timestep_i].tolist()
                    else:
                        self.config_kalib["objects"][0]["keypoints"][ptidx]["predicted_location"] = [-999.0, -999.0]

                json.dump(self.config_kalib, open(join(self.save_paths[camera], f"{timestep_i:06d}.json"), "w"))


    def calibrate_cameras(self):

        for camera in self.config.configured_cams:

            cam_data_dir = join(self.save_paths[camera])
            img_paths = sorted([join(cam_data_dir, f) for f in os.listdir(cam_data_dir) if f.endswith('.png')])
            json_paths = sorted([join(cam_data_dir, f) for f in os.listdir(cam_data_dir) if f.endswith('.json')])
            img_items = [cv2.imread(img_path) for img_path in img_paths]
            json_items = [json.load(open(json_path, 'r'))["objects"][0] for json_path in json_paths]
            
            robot_state_dicts = []
            
            for i, (img_item, json_item) in (pbar:=tqdm(enumerate(zip(img_items, json_items)), 
                                                        total=len(img_paths), 
                                                        desc=f"Parsing frames for {camera}")):
                pbar.update(1)
                robot_state = {
                    "local_to_world_matrix": json_item["local_to_world_matrix"],
                    "camera_intrinsics": json_item["camera_intrinsics"],
                    "img_obj": img_item,
                    "gripper_cartesian_position": np.asarray([kp_obj["location"] for kp_obj in json_item["keypoints"]])[0],           # 3d canonical frame coords for pnp
                    "inference_gripper_proj_loc": np.asarray([kp_obj["predicted_location"] for kp_obj in json_item["keypoints"]])[0], # 2d image plane coords for pnp
                    "gt_gripper_proj_loc": np.asarray([kp_obj["projected_location"] for kp_obj in json_item["keypoints"]])[0],        # warning: CANNOT trust this for now. Need good initial extrinsics
                    "joint_positions": json_item["joint_positions"],
                }
                robot_state_dicts.append(robot_state)
                
            word_to_local_mat = np.eye(4)
            
            points3d_canonical = np.stack([d['gripper_cartesian_position'] for d in robot_state_dicts]) # (num_frames,3)
            points2d_planar = np.stack([d['inference_gripper_proj_loc'] for d in robot_state_dicts])    # (num_frames,2)
            camera_intrinsics = np.stack([d['camera_intrinsics'] for d in robot_state_dicts])

            ret_val, translation, quaternion, reprojection_err = solve_pnp(points3d_canonical,
                                                                           points2d_planar,
                                                                           camera_intrinsics[0].copy()) # ! num_of_joints x 3, num_of_joints x 2, 3 x 3
            extrinsics_mat = rtvec_to_matrix(translation, quaternion)
            intrinsic_mat = camera_intrinsics[0].copy()
            
            # o3d.visualization.draw_geometries([
            #     o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points3d_canonical)),
            # ])
            
            for i, robot_state in (pbar:=tqdm(enumerate(robot_state_dicts), 
                                                        total=len(robot_state_dicts), 
                                                        desc=f"Rendering frames for {camera}")):
                pbar.update(1)
                img = robot_state["img_obj"]
                
                eef_coord_xyz = apply_se3_mat(extrinsics_mat,
                                              robot_state["gripper_cartesian_position"][None,...])
                
                eef_coord_xy = apply_intrinsics_mat(intrinsic_mat,
                                                    eef_coord_xyz)
                
                cv2.circle(img, tuple(eef_coord_xy[0].astype(int)), 5, (0, 0, 255), -1)
                cv2.imwrite(join(cam_data_dir, f"debug_reproj_calibrated_{i:06d}_{camera}.png"), img)
            
            
    
    


if __name__ == '__main__':

    # tf_data_loader = get_tf_dataloader(
    #     path="/mnt/nvme2n1_4t/data_stash/droid_data/droid_100/1.0.0/",
    #     batch_size=1,
    # )

    # "/mnt/nvme2n1_4t/data_stash/kalib_test_data/test_extract_droid_data"
    trajectory_dataset = TrajectoryDataset(data_folder_path=abspath('./assets/droid_examples/droid_Thu_May_11_13_33_20_2023'),
                                           recording_prefix="SVO",
                                           timestep_filtering_kwargs={"gripper_action_space": ["cartesian_position"]})
    calibrator = CalibratorCam2Base(trajectory_dataset, subsample_stride=10)
    calibrator.prepare_kalib_data()
    calibrator.calibrate_cameras()