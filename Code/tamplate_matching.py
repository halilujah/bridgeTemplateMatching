import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os

mesh3 = o3d.io.read_triangle_mesh("models/model_final.ply", enable_post_processing=True,print_progress=True)

pcd = mesh3.sample_points_uniformly(number_of_points=100000)

objectnames = [
    'P1',
    'BRRight1',
    'PierColumn5',
    'SIG4',
    'BentCap1'
]
meshes = [
    o3d.io.read_triangle_mesh(f"models/{objectnames[0]}.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[1]}.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[2]}.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[3]}.ply"),
    o3d.io.read_triangle_mesh(f"models/{objectnames[4]}.ply"),
]

for mesh in meshes:
    mesh.compute_vertex_normals()

pcd_points = np.asarray(pcd.points)

filtered_pcds = []  

for mesh in meshes:
    bbox = mesh.get_axis_aligned_bounding_box()
    
    cropped_pcd = pcd.crop(bbox)
    filtered_pcds.append(cropped_pcd)

# o3d.visualization.draw_geometries(filtered_pcds)


segmented_point_clouds = []  

for i, mesh in enumerate(meshes):
    mesh_vertices = np.asarray(mesh.vertices)  
    mesh_tree = cKDTree(mesh_vertices)  
    
    cropped_points = np.asarray(filtered_pcds[i].points)

    distances, indices = mesh_tree.query(cropped_points)

    segment_pcd = o3d.geometry.PointCloud()
    segment_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    segment_pcd.paint_uniform_color(np.random.rand(3))  

    filename = os.path.join("models/", f"{objectnames[i]}_segmented.ply") 
    o3d.io.write_point_cloud(filename, segment_pcd)
    segmented_point_clouds.append(segment_pcd)

o3d.visualization.draw_geometries(segmented_point_clouds)

