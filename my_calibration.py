import cv2, numpy as np
import pybullet as pb


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

class CalibrateCam2Base():
    
    def __init__(self, keypoint_inference_path:str='keypoint_inference_result.npz'):
        
        self.keypoint_inference_path = keypoint_inference_path
        self.keypoint_coords = np.load(keypoint_inference_path)['arr_0']
        
        


if __name__ == '__main__':
    
    calibration = CalibrateCam2Base(keypoint_inference_path='keypoint_inference_result.npz')