import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff

objectnames = [
    'P1',
    'BRRight1',
    'PierColumn5',
    'SIG4',
    'BentCap1'
]
segmented_point_clouds = [
    o3d.io.read_point_cloud(f"models/{objectnames[0]}_segmented.ply"),
    o3d.io.read_point_cloud(f"models/{objectnames[1]}_segmented.ply"),
    o3d.io.read_point_cloud(f"models/{objectnames[2]}_segmented.ply"),
    o3d.io.read_point_cloud(f"models/{objectnames[3]}_segmented.ply"),
    o3d.io.read_point_cloud(f"models/{objectnames[4]}_segmented.ply"),
]

meshes = [
    o3d.io.read_triangle_mesh(f"models/{objectnames[0]}_afterchange.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[1]}_afterchange.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[2]}_afterchange.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[3]}_afterchange.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[4]}_afterchange.ply"),
]

def align_point_cloud_to_mesh(pcd, mesh):
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=len(pcd.points))
    init_transformation = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, mesh_pcd, max_correspondence_distance=10,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return reg_p2p.transformation

def compute_translation_change(transformation_matrix):
    translation = transformation_matrix[:3, 3]
    return np.linalg.norm(translation)

def compute_rotation_change(transformation_matrix):
    rotation_matrix = np.array(transformation_matrix[:3, :3], dtype=np.float64)
    rotation = R.from_matrix(rotation_matrix)
    return np.degrees(rotation.magnitude())

def compute_vertex_based_centroid(mesh):
    return np.mean(np.asarray(mesh.vertices), axis=0)

def compute_centroid(pcd):
    return np.mean(np.asarray(pcd.points), axis=0)

def hausdorff_distance(pcd1, pcd2):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    return max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])

def compute_point_to_mesh_distances(pcd, mesh):
    mesh_vertices = np.asarray(mesh.vertices)
    tree = cKDTree(mesh_vertices)
    distances, _ = tree.query(np.asarray(pcd.points))
    return distances

POSITION_THRESHOLD = 0.05  
ROTATION_THRESHOLD = 5  

for i, name in enumerate(objectnames):
    print(f"Processing {name}...")
    
    transformation_matrix = align_point_cloud_to_mesh(segmented_point_clouds[i], meshes[i])
    position_change = compute_translation_change(transformation_matrix)
    rotation_change = compute_rotation_change(transformation_matrix)
    centroid_old = compute_vertex_based_centroid(meshes[i])
    centroid_new = compute_centroid(segmented_point_clouds[i])
    centroid_position_change = np.linalg.norm(centroid_new - centroid_old)
    mesh_pcd = meshes[i].sample_points_uniformly(number_of_points=len(segmented_point_clouds[i].points))
    hausdorff_dist = hausdorff_distance(mesh_pcd, segmented_point_clouds[i])
    distances = compute_point_to_mesh_distances(segmented_point_clouds[i], meshes[i])
    
    print(f"Position Change: {position_change*3.2808399:.4f} ft")
    print(f"Rotation Change: {rotation_change:.2f} degrees")
    print(f"Centroid Position Change: {centroid_position_change*3.2808399:.4f} ft")
    print(f"Hausdorff Distance: {hausdorff_dist*3.2808399:.4f} ft")
    print(f"Max Point-to-Mesh Distance: {max(distances)*3.2808399:.4f} ft, Mean: {np.mean(distances):.4f} ft")
    
    if position_change > POSITION_THRESHOLD or rotation_change > ROTATION_THRESHOLD:
        print("Object has moved significantly!")
    else:
        print("No significant movement detected.")
    
    # # Visualization of deviations
    # colors = np.zeros((len(distances), 3))
    # colors[:, 0] = distances / max(distances)  # Color intensity based on deviation
    # segmented_point_clouds[i].colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([segmented_point_clouds[i]])