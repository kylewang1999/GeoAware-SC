import os, sys, cv2, numpy as np, functools, yaml
import pybullet as pb
import pybullet_data
from os.path import dirname, abspath, join, exists
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from types import SimpleNamespace as NSpace


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

class TrajectorySampler:
    ''' Reference: https://github.com/robotflow-initiative/Kalib.git '''
    def __init__(
            self,
            all_folderpaths,
            recording_prefix="",
            traj_loading_kwargs={},
            timestep_filtering_kwargs={},
            image_transform_kwargs={},
            camera_kwargs={},
    ):
        self._all_folderpaths = all_folderpaths
        self.recording_prefix = recording_prefix
        self.traj_loading_kwargs = traj_loading_kwargs
        self.timestep_processer = TimestepProcesser(
            **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )
        self.camera_kwargs = camera_kwargs

    def fetch_samples(self, traj_ind: int = None):
        folderpath = self._all_folderpaths[traj_ind]

        filepath = os.path.join(folderpath, "trajectory.h5")
        recording_folderpath = os.path.join(folderpath, "recordings", self.recording_prefix)
        if not os.path.exists(recording_folderpath):
            recording_folderpath = None

        traj_samples = load_trajectory(
            filepath,
            recording_folderpath=recording_folderpath,
            camera_kwargs=self.camera_kwargs,
            **self.traj_loading_kwargs,
        )

        processed_traj_samples = [self.timestep_processer.forward(t, concat_states=False) for t in traj_samples]

        return processed_traj_samples, folderpath
    

class TrajectoryDataset():
    ''' Dataset class for one item in TrajectorySampler '''
    def __init__(self, trajectory_sampler, all_folderpaths):
        self._trajectory_sampler = trajectory_sampler
        self._trajectory_len = len(all_folderpaths)
        self._trajectory_cnt = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        # This will fetch and return the entire batch of samples from _refresh_generator
        if self._trajectory_cnt < self._trajectory_len:
            return self._refresh_generator()
        else:
            raise StopIteration
    def _refresh_generator(self):
        # Check if the trajectory count has reached the limit
        if self._trajectory_cnt >= self._trajectory_len:
            raise StopIteration

        # Fetch new samples and increment the trajectory counter
        timesteps, folder_path = self._trajectory_sampler.fetch_samples(traj_ind=self._trajectory_cnt)
        print("Examining the folderpath: ", folder_path)
        self._trajectory_cnt += 1

        # Return the new samples without creating a separate generator
        return folder_path, timesteps


class CalibratorCam2Base():
    
    def __init__(self, trajectory_dataset, keypoint_inference_path:str='keypoint_inference_result.npz'):
        
        self.keypoint_inference_path = keypoint_inference_path
        self.keypoint_coords = np.load(keypoint_inference_path)['arr_0']
        self.trajectory_dataset = trajectory_dataset
        
        self.configured_cams = ["varied_camera_1_left", "varied_camera_1_right",
                                "varied_camera_2_left", "varied_camera_2_right"]
        with open('./configs/calibration_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        self.config = dict2namespace(config)
        
        if not exists(self.config.output_dir): os.makedirs(self.config.output_dir)
        self.save_paths = {key: join(self.config.output_dir, key) for key in self.configured_cams}
        self.save_imgs = {key: [] for key in self.configured_cams}
        self.debug_save_imgs = {key: [] for key in self.configured_cams}

        
    def kalibify_timestep(self, timestep):
        ''' Convert droid timestep to kalib format '''
        
        extrinsics_dict = timestep["extrinsics_dict"]
        intrinsics_dict = timestep["intrinsics_dict"]
        # import pdb; pdb.set_trace()
        # processed_timestep["observation"] contains keys: 'cartesian_position', 'gripper_position', 'joint_positions', 'joint_torques_computed', 'joint_velocities', 'motor_torques_measured', 'prev_command_successful', 'prev_controller_latency_ms', 'prev_joint_torques_computed', 'prev_joint_torques_computed_safened'

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
            "camera/intrinsics/varied_camera_2_right": intrinsics_dict["varied_camera"][3],
    }

        for k in obs: obs[k] = np.array(obs[k])

        return obs

    
    def calibrate(self):
        
        folder_path, timesteps = next(iter(self.trajectory_dataset))
    
        for timestep_i, timestep in (pbar:=tqdm(enumerate(timesteps))):

            timestep = self.kalibify_timestep(timestep)

            joint_positions = timestep["robot_state/joint_positions"]
            raw_eef_pos = timestep["robot_state/cartesian_position"]
            
            eefpos_from_joint_positions = compute_forward_kinematics(urdf_file=self.config.urdf_path, 
                                                                     joint_positions=joint_positions)
            eefpos_from_joint_positions = eefpos_from_joint_positions[-1]
            eefpos = np.array(eefpos_from_joint_positions)
            robot_joints = eefpos[None]

            # Forward every camera: available options: 'varied_camera_1_left', 'varied_camera_1_right', 'varied_camera_2_left', 'varied_camera_2_right'
            for camera in self.configured_cams:
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
                debug_img = np.copy(self.save_imgs[camera])
                debug_img = cv2.circle(debug_img, tuple(projected_eefpos.astype(int)), 5, (0, 0, 255), -1)
                debug_img = cv2.circle(debug_img, tuple(projected_raw_eefpos.astype(int)), 5, (0, 255, 255), -1)

                self.debug_save_imgs[camera] = debug_img

                cv2.imwrite(join(self.save_paths[camera], f"{timestep_i:06d}_{camera}.png"), self.save_imgs[camera])
                cv2.imwrite(join(self.save_paths[camera], f"debug/{timestep_i:06d}_{camera}.png"), self.debug_save_imgs[camera])

                ROBOT_DATA["objects"][0]["eef_pos"] = eefpos.tolist()
                ROBOT_DATA["objects"][0]["camera_intrinsics"] = camera_intrinsics.tolist()
                ROBOT_DATA["objects"][0]["local_to_world_matrix"] = camera_extrinsics.tolist()
                ROBOT_DATA["objects"][0]["joint_positions"] = processed_timestep[
                    "robot_state/joint_positions"
                ].tolist()
                ROBOT_DATA["objects"][0]["cartesian_position"] = processed_timestep[
                    "robot_state/cartesian_position"
                ].tolist()

                for ptidx, pt in enumerate(robot_joints):
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["location"] = eefpos.tolist()
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["name"] = "panda_left_finger"
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["projected_location"] = projected_eefpos.tolist()
                    ROBOT_DATA["objects"][0]["keypoints"][ptidx]["predicted_location"] = [-999.0, -999.0]

                json.dump(ROBOT_DATA, open(osp.join(save_paths[camera], f"{ind:06d}.json"), "w"))



if __name__ == '__main__':

    # tf_data_loader = get_tf_dataloader(
    #     path="/mnt/nvme2n1_4t/data_stash/droid_data/droid_100/1.0.0/",
    #     batch_size=1,
    # )
    
    trajectory_sampler = TrajectorySampler(
        # all_folderpaths=[abspath('./assets/droid_examples/droid_Thu_May_11_13_33_20_2023')], 
        all_folderpaths=["/mnt/nvme2n1_4t/data_stash/kalib_test_data/test_extract_droid_data"], 
        recording_prefix="SVO",
        timestep_filtering_kwargs={"gripper_action_space": ["cartesian_position"]}
    )
    
    trajectory_dataset = TrajectoryDataset(
        trajectory_sampler, 
        all_folderpaths=["/mnt/nvme2n1_4t/data_stash/kalib_test_data/test_extract_droid_data"]
    )
    
    calibrator = CalibratorCam2Base(trajectory_dataset, keypoint_inference_path='keypoint_inference_result.npz')
    calibrator.calibrate()

    
    
    
    # calibration = CalibrateCam2Base(keypoint_inference_path='keypoint_inference_result.npz')