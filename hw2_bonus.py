from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import pickle
from itertools import permutations

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


# ---- helper quartic solvers ----
def _solve_cubic_real(coeffs):
    """Return real roots of cubic, coeffs = [a3,a2,a1,a0] (highest -> constant)."""
    roots = np.roots(coeffs)
    real_roots = [np.real(r) for r in roots if abs(np.imag(r)) < 1e-8]
    return real_roots

    
def _quartic_classical_mine(a4, a3, a2, a1, a0):

    if abs(a4) < 1e-16:
        return []
    b = a3 / a4
    c = a2 / a4
    d = a1 / a4
    e = a0 / a4

    # Depressed quartic: x = y - b/4
    p = c - 3*b*b/8
    q = d + b*b*b/8 - b*c/2
    r = e - 3*b**4/256 + (b*b*c)/16 - (b*d)/4

    # Solve resolvent cubic for z: u^3 + (5p/2) u^2 + (2r - 0.5p^2) u - (q^2)/8 = 0
    cubic_coeffs = [
        1.0,
        2.5*p,
        2.0*r - 0.5*p*p,
        - (q*q) / 8.0
    ]


    # compute two quadratic factors
    try:
        zs = _solve_cubic_real(cubic_coeffs)
        if len(zs) == 0:
            return []
        # pick the largest real z (heuristic, usually gives stable square roots)
        z = max(zs)

        if z < -1e-12:
            # If z slightly negative due to numerics, clamp to zero for sqrt
            z = 0.0
        sqrt_term = np.sqrt(2*z)
        if sqrt_term == 0:
            # fallback to np.roots later
            return []

        # Build two quadratics: y^2 +/- sqrt(2z) y + (p/2 + z +/- q/(2*sqrt(2z))) = 0
        q1 = sqrt_term
        A1 = 1.0
        B1 = q1
        C1 = p/2.0 + z - (q / (2.0*sqrt_term)) if abs(sqrt_term) > 1e-14 else p/2.0 + z

        A2 = 1.0
        B2 = -q1
        C2 = z + 0.5*p + (q / (2.0*sqrt_term)) if abs(sqrt_term) > 1e-14 else z + 0.5*p

        roots = []
        for A, B, C in ((A1,B1,C1), (A2,B2,C2)):
            disc = B*B - 4*A*C
            if disc >= -1e-12:
                disc = max(disc, 0.0)
                r1 = (-B + np.sqrt(disc)) / (2*A)
                r2 = (-B - np.sqrt(disc)) / (2*A)
                roots.extend([r1 - b/4.0, r2 - b/4.0])
        # filter reals
        real_roots = [float(r) for r in roots if abs(np.imag(r)) < 1e-8]
        return real_roots
    except Exception:
        return []

    
def _quartic_ferrari_lagrange_mine(a4, a3, a2, a1, a0):

    if abs(a4) < 1e-16:
        return []
    b = a3 / a4
    c = a2 / a4
    d = a1 / a4
    e = a0 / a4

    # Depressed transformation x = y - b/4
    p = c - 3*b*b/8
    q = d + b*b*b/8 - b*c/2
    r = e - 3*b**4/256 + (b*b*c)/16 - (b*d)/4

    # Ferrari-Lagrange resolvent cubic (alternative formulation)
    # We solve for u in: u^3 + (2*p) u^2 + (p^2 - 4r) u - q^2 = 0
    try:
        cubic_coeffs = [
            1.0,
            2.0 * p,
            p*p - 4*r,
            -q*q
        ]
        us = _solve_cubic_real(cubic_coeffs)
        if len(us) == 0:
            return []
        u = max(us)

        # compute values for quadratic solving
        alpha = np.sqrt(max(0.0, u))
        # construct quadratic factors based on Lagrange formulation:
        # y^2 +/- alpha*y + ( (p/2) -/+ (q/(2*alpha)) + (u/2) ) = 0
        roots = []
        for sign in (+1.0, -1.0):
            A = 1.0
            B = sign * alpha
            C = 0.5*p + 0.5*u - sign * (q/(2.0*alpha)) if abs(alpha) > 1e-12 else 0.5*p + 0.5*u # to prevent (q/(2.0*alpha)) blowup
            disc = B*B - 4*A*C
            if disc >= -1e-12:
                disc = max(disc, 0.0)
                r1 = (-B + np.sqrt(disc)) / (2*A)
                r2 = (-B - np.sqrt(disc)) / (2*A)
                roots.extend([r1 - b/4.0, r2 - b/4.0])
        real_roots = [float(rr) for rr in roots if abs(np.imag(rr)) < 1e-8]
        return real_roots
    except Exception:
        return []

def _solve_quartic_switch(coeffs, use_lagrange=False):
    a4, a3, a2, a1, a0 = coeffs
    roots = []
    try:
        if use_lagrange:
            roots = _quartic_ferrari_lagrange_mine(a4, a3, a2, a1, a0)
        else:
            roots = _quartic_classical_mine(a4, a3, a2, a1, a0)

        # If analytic solver produced few/no roots, fallback to numpy.roots
        if len(roots) == 0:
            # print("Quartic solver produced no roots, falling back to np.roots")
            poly = np.array([a4, a3, a2, a1, a0], dtype=np.complex128)
            rts = np.roots(poly)
            roots = [np.real(r) for r in rts if abs(np.imag(r)) < 1e-8]
    except Exception:
        # print("fallback np.roots()")
        poly = np.array([a4, a3, a2, a1, a0], dtype=np.complex128)
        rts = np.roots(poly)
        roots = [np.real(r) for r in rts if abs(np.imag(r)) < 1e-8]

    # keep only strictly positive roots (threshold)
    pos_roots = [float(r) for r in roots if r > 1e-8]
    # remove duplicates within tolerance
    unique = []
    for r in pos_roots:
        if not any(abs(r - u) < 1e-6 for u in unique):
            unique.append(r)
    return unique


def solve_p3p(X, m):

    # Step 1: Compute dot products between unit vectors
    m12 = float(np.dot(m[0], m[1]))
    m13 = float(np.dot(m[0], m[2]))
    m23 = float(np.dot(m[1], m[2]))

    # Step 2: Reindex points to ensure m13 <= m12 <= m23
    # We want order of pairs to be [m13, m12, m23] corresponding to sorted list by paper
    # Find permutation of indices such that the pair-order becomes (0,2),(0,1),(1,2)
    # m13 <= m12 <= m23 after permuting
    perm_found = None
    for perm in permutations([0,1,2]):
        m_perm = [np.dot(m[perm[i]], m[perm[j]]) for i,j in [(0,1),(0,2),(1,2)]]
        # m_perm corresponds to m12', m13', m23' under this ordering
        if m_perm[1] <= m_perm[0] <= m_perm[2] + 1e-12:
            perm_found = perm
            break
    if perm_found is not None:
        if list(perm_found) != [0,1,2]:
            X = X[list(perm_found)]
            m = m[list(perm_found)]
            m12 = float(np.dot(m[0], m[1]))
            m13 = float(np.dot(m[0], m[2]))
            m23 = float(np.dot(m[1], m[2]))
    # else: leave as-is (degenerate or already ordered)

    # Step 3: Compute squared distances between 3D points
    s12 = float(np.sum((X[0] - X[1])**2))
    s13 = float(np.sum((X[0] - X[2])**2))
    s23 = float(np.sum((X[1] - X[2])**2))

    # Step 4: Compute coefficients A, B, C (equations 23-25)
    A = -s12 + s23 + s13
    B = 2 * (s12 - s23) * m13
    C = -s12 + s23 - s13

    # Step 5: Compute quartic coefficients (equations 27-31)
    c4 = (-s12**2 + 2*s12*s13 + 2*s12*s23 - s13**2 + 4*s13*s23*m12**2 -
          2*s13*s23 - s23**2)

    c3 = (4*s12**2*m13 - 4*s12*s13*m12*m23 - 4*s12*s13*m13 - 8*s12*s23*m13 +
          4*s13**2*m12*m23 - 8*s13*s23*m12**2*m13 - 4*s13*s23*m12*m23 +
          4*s13*s23*m13 + 4*s23**2*m13)

    c2 = (-4*s12**2*m13**2 - 2*s12**2 + 8*s12*s13*m12*m13*m23 + 4*s12*s13*m23**2 +
          8*s12*s23*m13**2 + 4*s12*s23 - 4*s13**2*m12**2 - 4*s13**2*m23**2 +
          2*s13**2 + 4*s13*s23*m12**2 + 8*s13*s23*m12*m13*m23 - 4*s23**2*m13**2 -
          2*s23**2)

    c1 = (4*s12**2*m13 - 4*s12*s13*m12*m23 - 8*s12*s13*m13*m23**2 + 4*s12*s13*m13 -
          8*s12*s23*m13 + 4*s13**2*m12*m23 - 4*s13*s23*m12*m23 - 4*s13*s23*m13 +
          4*s23**2*m13)

    c0 = (-s12**2 + 4*s12*s13*m23**2 - 2*s12*s13 + 2*s12*s23 - s13**2 +
          2*s13*s23 - s23**2)

    # Step 6: Solve quartic equation
    if abs(c4) < 1e-14:
        return []  # Degenerate case

    coeffs = [c4, c3, c2, c1, c0]

    # paper criterion: use Ferrari-Lagrange if |c3/c4| > 10
    use_lagrange = abs(c3 / c4) > 10.0

    # Solve quartic robustly, accept only positive real roots
    x_roots = _solve_quartic_switch(coeffs, use_lagrange)

    solutions = []

    # Step 7: For each valid x, compute y and depths
    for x in x_roots:
        # Compute y using equation 22
        denominator = 2.0 * s13 * (m12 * x - m23)
        if abs(denominator) < 1e-12:
            continue

        y = (A * x**2 + B * x + C) / denominator
        if not (y > 1e-8):
            continue

        # Compute d3 using equation 16
        denom = y**2 - 2.0*y*m23 + 1.0
        if denom <= 1e-12:
            continue
        d3 = np.sqrt(s23 / denom)
        d1 = x * d3
        d2 = y * d3

        if d1 <= 1e-8 or d2 <= 1e-8 or d3 <= 1e-8:
            continue

        # GN refinement on depths
        depths = np.array([d1, d2, d3], dtype=np.float64)
        for _ in range(5):
            d1, d2, d3 = depths
            r1 = d1**2 + d2**2 - 2*d1*d2*m12 - s12
            r2 = d1**2 + d3**2 - 2*d1*d3*m13 - s13
            r3 = d2**2 + d3**2 - 2*d2*d3*m23 - s23
            if abs(r1) + abs(r2) + abs(r3) < 1e-9:
                break

            J = np.array([
                [2*d1 - 2*d2*m12, 2*d2 - 2*d1*m12, 0.0],
                [2*d1 - 2*d3*m13, 0.0, 2*d3 - 2*d1*m13],
                [0.0, 2*d2 - 2*d3*m23, 2*d3 - 2*d2*m23]
            ], dtype=np.float64)
            residuals = np.array([r1, r2, r3], dtype=np.float64)

            try:
                delta = np.linalg.solve(J.T @ J + 1e-8 * np.eye(3), -J.T @ residuals)
            except np.linalg.LinAlgError:
                break
            depths = depths + delta
            if np.any(depths <= 1e-8):
                break

        d1, d2, d3 = depths
        if d1 <= 1e-8 or d2 <= 1e-8 or d3 <= 1e-8:
            continue

        # Step 9: compute R,t
        P_cam = np.array([d1*m[0], d2*m[1], d3*m[2]], dtype=np.float64)

        Y1 = P_cam[0] - P_cam[1]
        Y2 = P_cam[0] - P_cam[2]
        Y3 = np.cross(Y1, Y2)

        X1_minus_X2 = X[0] - X[1]
        X1_minus_X3 = X[0] - X[2]
        X3 = np.cross(X1_minus_X2, X1_minus_X3)

        Y_matrix = np.column_stack([Y1, Y2, Y3])
        X_matrix = np.column_stack([X1_minus_X2, X1_minus_X3, X3])

        try:
            X_inv = np.linalg.inv(X_matrix)
            R_matrix = Y_matrix @ X_inv
            t_vector = P_cam[0] - R_matrix @ X[0]

            # Ensure R is a proper rotation matrix via SVD (Orthogonal Procrustes solution.)
            U, _, Vt = np.linalg.svd(R_matrix)
            R_matrix = U @ Vt
            if np.linalg.det(R_matrix) < 0:
                U[:, -1] *= -1
                R_matrix = U @ Vt

            solutions.append((R_matrix, t_vector))
        except np.linalg.LinAlgError:
            continue

    return solutions


def pnpsolver(query, model, cameraMatrix=None, distortion=None):
    """
    Custom P3P+RANSAC solver
    """
    kp_query, desc_query = query
    kp_model, desc_model = model
    
    if cameraMatrix is None:
        cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    if distortion is None:
        distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    
    # Descriptor matching using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc_query, desc_model, k=2)
    
    # Apply ratio test
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
    
    # Undistort points first
    query_pts_undist = cv2.undistortPoints(
        query_pts.reshape(-1, 1, 2), 
        cameraMatrix, 
        distCoeffs
    ).reshape(-1, 2)
    
    # Convert to unit bearing vectors
    bearing_vectors = []
    for pt in query_pts_undist:
        # Normalized image coordinates to 3D bearing vector
        bearing = np.array([pt[0], pt[1], 1.0])
        bearing = bearing / np.linalg.norm(bearing)
        bearing_vectors.append(bearing)
    bearing_vectors = np.array(bearing_vectors)


    ## self add early stopping RANSAC
    max_iterations = 1000          # hard cap (safety)
    desired_prob = 0.99            # success probability
    s = 3                          # sample size for P3P

    inlier_threshold = 8.0
    best_inliers = []
    best_inliers_len = -1
    best_R = None
    best_t = None

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        sample_indices = np.random.choice(len(good_matches), 3, replace=False)
        X_sample = model_pts[sample_indices]
        m_sample = bearing_vectors[sample_indices]

        solutions = solve_p3p(X_sample, m_sample)

        for R_sol, t_sol in solutions:
            projected_pts = []
            for X_pt in model_pts:
                X_cam = R_sol @ X_pt + t_sol
                if X_cam[2] <= 0:
                    break

                x_proj = X_cam[0] / X_cam[2]
                y_proj = X_cam[1] / X_cam[2]

                pts_normalized = np.array([[x_proj, y_proj, 1.0]], dtype=np.float32)
                pts_distorted, _ = cv2.projectPoints(
                    pts_normalized.reshape(-1, 1, 3),
                    np.zeros(3), np.zeros(3),
                    cameraMatrix, distCoeffs
                )
                projected_pts.append(pts_distorted[0, 0])

            if len(projected_pts) != len(model_pts):
                continue

            projected_pts = np.array(projected_pts)
            errors = np.linalg.norm(query_pts - projected_pts, axis=1)
            inliers = np.where(errors < inlier_threshold)[0]

            if len(inliers) > best_inliers_len:
                best_inliers = inliers
                best_inliers_len = len(inliers)
                best_R = R_sol.copy()
                best_t = t_sol.copy()

                w = len(best_inliers) / len(good_matches)
                if 1 > w > 0:
                    N = np.log(1 - desired_prob) / np.log(1 - (w ** s))
                    if iteration >= N:
                        iteration = max_iterations
                        break
                elif w == 1:
                    iteration = max_iterations
                    break
 
    # Convert rotation matrix to rotation vector
    rvec = R.from_matrix(best_R).as_rotvec()
    
    return True, rvec.reshape(3, 1), best_t.reshape(3, 1), best_inliers

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
    # # Load data
    print("Loading data...")
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # print(images_df['NAME'].unique())
    # print(images_df['IMAGE_ID'].unique())
    
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

    # print(points3D_df.head())
    
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

    with open("results_bonus.txt", "w", encoding="utf-8") as f:
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
    video_writer = cv2.VideoWriter('ar_cube_video_bonus.mp4', fourcc, 10.0, (w, h))
    

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
            
            sorted_indices = painters_algorithm_sort(cube_points_cam)
            
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
    
    print(f"AR video saved to: ar_cube_video_bonus.mp4")