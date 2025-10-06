from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query, model, cameraMatrix=0, distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # Descriptor matching using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches
    matches = flann.knnMatch(desc_query, desc_model, k=2)
    
    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return False, None, None, None
    
    # Extract matched points
    query_pts = np.array([kp_query[m.queryIdx] for m in good_matches])
    model_pts = np.array([kp_model[m.trainIdx] for m in good_matches])
    
    # Solve PnP problem
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        model_pts.reshape(-1, 1, 3),
        query_pts.reshape(-1, 1, 2),
        cameraMatrix,
        distCoeffs,
        reprojectionError=8.0,
        confidence=0.99
    )
    
    return retval, rvec, tvec, inliers

def rotation_error(R1, R2):
    # Convert quaternions to rotation objects
    rot1 = R.from_quat(R1)  # Ground truth
    rot2 = R.from_quat(R2)  # Estimation
    
    # Calculate relative rotation: R_rel = R_gt^(-1) * R_est
    R_rel = rot1.inv() * rot2
    
    # Convert to axis-angle representation and get the angle
    rotvec = R_rel.as_rotvec()
    angle = np.linalg.norm(rotvec)  # The magnitude is the rotation angle
    
    return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

def create_camera_pyramid(pose, size=0.3, color=[1, 0, 0]):
    # Camera pyramid vertices in camera coordinate system
    # Apex at origin (optical center)
    apex = np.array([0, 0, 0])
    
    base_vertices = np.array([
        [-size, -size, size],   # bottom-left
        [ size, -size, size],   # bottom-right
        [ size,  size, size],   # top-right
        [-size,  size, size]    # top-left
    ])
    
    vertices = np.vstack([apex, base_vertices])
    
    vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_world = (pose @ vertices_homogeneous.T).T[:, :3]
    
    # Define edges connecting apex to base vertices and base edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # apex to base vertices
        [1, 2], [2, 3], [3, 4], [4, 1]   # base edges
    ]
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return line_set

def create_trajectory_line(poses, color=[0, 1, 0]):

    positions = []
    for pose in poses:
        if pose is not None:
            positions.append(pose[:3, 3])
    
    if len(positions) < 2:
        return None
    
    positions = np.array(positions)
    lines = [[i, i+1] for i in range(len(positions)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return line_set

def visualization_open3d(Camera2World_Transform_Matrixs, points3D_df):
    print("Creating Open3D visualization...")
    
    # Create point cloud from 3D points
    points_3d = np.array(points3D_df["XYZ"].to_list())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    colors = np.vstack(points3D_df['RGB'].values)
    colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Create camera pyramids
    valid_poses = []
    for i, pose in enumerate(Camera2World_Transform_Matrixs):
        if pose is not None:
            # Create camera pyramid (red color)
            camera_pyramid = create_camera_pyramid(pose, size=0.2, color=[1, 0, 0])
            geometries.append(camera_pyramid)
            valid_poses.append(pose)
            
    
    # Create trajectory line
    if len(valid_poses) > 1:
        trajectory = create_trajectory_line(valid_poses, color=[0, 1, 0])
        if trajectory is not None:
            geometries.append(trajectory)

    # flip and fix origin
    flip_transform = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ])
    for g in geometries:
        g.transform(flip_transform)

    Ry_small = np.eye(4)
    angle = np.deg2rad(-9)  # adjust the tilt
    Ry_small[0,0] = np.cos(angle)
    Ry_small[0,2] = np.sin(angle)
    Ry_small[2,0] = -np.sin(angle)
    Ry_small[2,2] = np.cos(angle)

    for g in geometries:
        g.transform(Ry_small)

    
    # Visualize
    print(f"Visualizing {len(points_3d)} 3D points and {len(valid_poses)} camera poses")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Relocalization - Poses and 3D Points",
        width=1200,
        height=800
    )


def create_cube_voxels(size=1.0, voxels_per_edge=10):
    points = []
    colors = []
    step = size / voxels_per_edge
    
    face_colors = {
        'front': [255, 0, 0],      # Red
        'back': [0, 255, 0],       # Green
        'left': [0, 0, 255],       # Blue
        'right': [255, 255, 0],    # Yellow
        'top': [255, 0, 255],      # Magenta
        'bottom': [0, 255, 255]    # Cyan
    }
    
    for i in range(voxels_per_edge):
        for j in range(voxels_per_edge):
            x = i * step
            y = j * step
            
            points.extend([
                [x, y, size],      # Front (z = size)
                [x, y, 0],         # Back (z = 0)
                [0, x, y],         # Left (x = 0)
                [size, x, y],      # Right (x = size)
                [x, size, y],      # Top (y = size)
                [x, 0, y]          # Bottom (y = 0)
            ])
            
            colors.extend([
                face_colors['front'],
                face_colors['back'],
                face_colors['left'],
                face_colors['right'],
                face_colors['top'],
                face_colors['bottom']
            ])
    
    return np.array(points), np.array(colors)


def apply_transform_to_cube(cube_points, transform_mat):
    ones = np.ones((cube_points.shape[0], 1))
    points_homogeneous = np.hstack([cube_points, ones])
    return (transform_mat @ points_homogeneous.T).T


def painters_algorithm_sort(points_3d_cam):
    """Sort points by depth (furthest first) for painter's algorithm."""
    return np.argsort(-points_3d_cam[:, 2])


def project_points_to_image(points_3d_cam, camera_matrix, dist_coeffs):
    """Project 3D points to 2D image plane with distortion."""
    # valid_mask = points_3d_cam[:, 2] > 0
    valid_mask = (points_3d_cam[:, 2] > 0.05) & (points_3d_cam[:, 2] < 100.0)
    
    if not np.any(valid_mask):
        return np.array([]), valid_mask
    
    points_3d_valid = points_3d_cam[valid_mask].reshape(-1, 1, 3)
    points_2d_valid, _ = cv2.projectPoints(
        points_3d_valid, np.zeros(3), np.zeros(3),
        camera_matrix, dist_coeffs
    )
    
    return points_2d_valid.reshape(-1, 2), valid_mask


def draw_cube_on_image(image, points_2d, colors, point_size=5):
    """Draw colored voxel points on image."""
    h, w = image.shape[:2]
    
    for pt, color in zip(points_2d, colors):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), point_size,
                      (int(color[2]), int(color[1]), int(color[0])), -1)
    
    return image



def is_horizontal_scatter(points_2d, img_w, img_h, idx, fname):
    if points_2d is None or len(points_2d) < 20:
        return False

    x = np.clip(points_2d[:,0], -img_w*2, img_w*3)
    y = np.clip(points_2d[:,1], -img_h*2, img_h*3)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    x_norm = x_range / img_w
    y_norm = y_range / img_h

    # print(idx, fname, x_norm, y_norm)

    # Condition: wide in X but flat in Y
    if x_norm > 2:
        return True
    return False


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    print("Processing model descriptors...")
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    IMAGE_ID_LIST = [i for i in range(164,294)] #[146, 201, 240, 280]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []

    print(points3D_df.head())
    
    print("Processing query images...")
    for idx in tqdm(IMAGE_ID_LIST):
        # Load query image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        
        # Convert rotation vector to quaternion
        rotq = R.from_rotvec(rvec.reshape(3)).as_quat()  # [x, y, z, w]
        tvec = tvec.reshape(3)
        r_list.append(rotq)
        t_list.append(tvec)
        
        # Get camera pose groundtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
        rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values[0]  # [x, y, z, w]
        tvec_gt = ground_truth[["TX", "TY", "TZ"]].values[0]
        
        # Calculate rotation error using relative rotation and axis-angle
        r_error = rotation_error(rotq_gt, rotq)
        t_error = translation_error(tvec_gt, tvec)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)
        
        # print(f"Image {idx}: Rotation error = {r_error:.4f}°, Translation error = {t_error:.4f}")


    # Calculate median of relative rotation angle differences and translation differences
    valid_rotation_errors = [r for r in rotation_error_list if r != float('inf')]
    valid_translation_errors = [t for t in translation_error_list if t != float('inf')]
    
    print("\n" + "="*50)
    print("RESULTS:")

    median_rotation_error = np.median(valid_rotation_errors)
    mean_rotation_error = np.mean(valid_rotation_errors)
    print(f"Median rotation error: {median_rotation_error:.4f} degrees")
    print(f"Mean rotation error: {mean_rotation_error:.4f} degrees")
    print(f"Individual rotation errors: {[f'{r:.4f}°' for r in valid_rotation_errors]}")

    median_translation_error = np.median(valid_translation_errors)
    mean_translation_error = np.mean(valid_translation_errors)
    print(f"Median translation error: {median_translation_error:.4f}")
    print(f"Mean translation error: {mean_translation_error:.4f}")
    print(f"Individual translation errors: {[f'{t:.4f}' for t in valid_translation_errors]}")

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("RESULTS:\n")
        f.write(f"Median rotation error: {median_rotation_error:.4f} degrees\n")
        f.write(f"Mean rotation error: {mean_rotation_error:.4f} degrees\n")
        f.write(f"Individual rotation errors: {[f'{r:.4f}°' for r in valid_rotation_errors]}\n")
        f.write(f"Median translation error: {median_translation_error:.4f}\n")
        f.write(f"Mean translation error: {mean_translation_error:.4f}\n")
        f.write(f"Individual translation errors: {[f'{t:.4f}' for t in valid_translation_errors]}\n")


    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # PnP gives us world-to-camera transformation, convert to camera-to-world
        R_w2c = R.from_quat(r).as_matrix()
        t_w2c = t
        
        # Camera-to-world transformation
        R_c2w = R_w2c.T  # Inverse of rotation
        t_c2w = -R_c2w @ t_w2c  # Inverse transformation for translation
        
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        Camera2World_Transform_Matrixs.append(c2w)
    
    # Visualization with Open3D
    visualization_open3d(Camera2World_Transform_Matrixs, points3D_df)


    print("\n" + "="*60)
    print("CREATING AR VIDEO WITH VIRTUAL CUBE")
    print("="*60)
    
    cube_transform_mat = np.load('cube_transform_mat.npy')
    
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]])
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])
    
    print("\nCreating cube voxels (10x10x10 resolution)...")
    cube_points, cube_colors = create_cube_voxels(size=1.0, voxels_per_edge=10)
    cube_points_world = apply_transform_to_cube(cube_points, cube_transform_mat)
    
    images_subset = images_df[images_df["IMAGE_ID"].isin(IMAGE_ID_LIST)].copy()
    
    # Extract numeric part from NAME (e.g., "valid_img85.png" -> 85)
    images_subset['img_number'] = images_subset['NAME'].str.extract(r'valid_img(\d+)\.jpg')[0].astype(int)
    images_subset = images_subset.sort_values("img_number").reset_index(drop=True)
    
    pose_dict = {}
    for idx, rotq, tvec in zip(IMAGE_ID_LIST, r_list, t_list):
        pose_dict[idx] = (rotq, tvec)
    
    first_fname = images_subset.iloc[0]["NAME"]
    first_image = cv2.imread("data/frames/" + first_fname)
    h, w = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('ar_cube_video_pnp.mp4', fourcc, 10.0, (w, h))
    

    for i, row in tqdm(images_subset.iterrows(), 
                       total=len(images_subset),
                       desc="Creating AR video"):
        idx = row["IMAGE_ID"]
        fname = row["NAME"]
        
        rotq, tvec = pose_dict.get(idx, (None, None))

        image = cv2.imread("data/frames/" + fname)
        
        if image is None:
            video_writer.write(first_image)
            continue
        
        if rotq is not None and tvec is not None:
            # Get world-to-camera transformation
            R_mat = R.from_quat(rotq).as_matrix()
            transform_w2c = np.hstack([R_mat, tvec.reshape(3, 1)])
            
            # Transform cube to camera coordinates
            cube_homogeneous = np.hstack([
                cube_points_world,
                np.ones((cube_points_world.shape[0], 1))
            ])
            cube_points_cam = (transform_w2c @ cube_homogeneous.T).T
            
            sorted_indices = np.argsort(-cube_points_cam[:, 2]) #painters_algorithm_sort(cube_points_cam)
            
            points_2d, valid_mask = project_points_to_image(
                cube_points_cam[sorted_indices],
                cameraMatrix,
                distCoeffs
            )


            if is_horizontal_scatter(points_2d, w, h, idx, fname):
                video_writer.write(image)
                continue
            
            # Draw cube
            if len(points_2d) > 0:
                sorted_colors = cube_colors[sorted_indices][valid_mask]
                image = draw_cube_on_image(image, points_2d, sorted_colors, point_size=3)

        
        video_writer.write(image)
    
    video_writer.release()
    
    print(f"AR video saved to: ar_cube_video_pnp.mp4")