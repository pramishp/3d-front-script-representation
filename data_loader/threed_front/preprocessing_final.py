import trimesh
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
from itertools import combinations
import pdb  # Import the debugger
import warnings
import gc
from scipy.spatial import ConvexHull
from datetime import datetime
import re
import logging

import sys
sys.path.append("/home/ajad/Desktop/codes/LISA")
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront

#-----------FURNITURE AND WALL COLLISION DETECTION-------------
# Checks if a point is within a bounding box with a small tolerance.
def is_point_in_bbox(point, bbox_min, bbox_max, tolerance=1e-6):
    return all(bbox_min[i] - tolerance <= point[i] <= bbox_max[i] + tolerance for i in range(3))

# Calculates the percentage of vertices of mesh1 inside the bounding box of mesh2.
def calculate_vertex_enclosure_score(mesh1, mesh2):
    bbox_min2, bbox_max2 = mesh2.bounding_box.bounds
    enclosed_count = sum(is_point_in_bbox(vertex, bbox_min2, bbox_max2) for vertex in mesh1.vertices)
    return enclosed_count / len(mesh1.vertices) if mesh1.vertices.size > 0 else 0

# Filters out furniture that is not sufficiently enclosed by the walls.
def filter_furniture_in_scene(wall_meshes, furniture_meshes, enclosure_threshold=0.5):
    # Combine wall meshes into one large mesh (to save memory)
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for wall_mesh in wall_meshes:
        combined_vertices.append(wall_mesh.vertices)
        combined_faces.append(wall_mesh.faces + vertex_offset)
        vertex_offset += len(wall_mesh.vertices)

    if not combined_vertices:
        return [], list(range(len(furniture_meshes)))  # No walls, all furniture considered invalid

    # Create the combined wall mesh
    combined_wall_mesh = trimesh.Trimesh(
        vertices=np.vstack(combined_vertices),
        faces=np.vstack(combined_faces),
        process=False
    )

    # Check each furniture mesh
    valid_furniture = []
    invalid_indices = []

    for i, furniture_mesh in enumerate(furniture_meshes):
        try:
            mesh = trimesh.Trimesh(vertices=furniture_mesh.vertices, faces=furniture_mesh.faces, process=False)
            score = calculate_vertex_enclosure_score(mesh, combined_wall_mesh)

            if score >= enclosure_threshold:
                valid_furniture.append(furniture_mesh)
            else:
                invalid_indices.append(i)
                logger.warning(f"⚠️ warning: Furniture {i} collides with walls, enclosure score: {score:.2f}")
        except Exception as e:
            logger.error(f"⚠️ Error processing furniture mesh {i}: {e}")
            invalid_indices.append(i)
            continue

    del combined_wall_mesh
    gc.collect()
    return valid_furniture, invalid_indices

def bbox_overlap(bbox_1, bbox_2):
    # Convert to numpy arrays for easier manipulation
    bbox_1 = np.array(bbox_1)
    bbox_2 = np.array(bbox_2)
    
    # Find min and max coordinates for each dimension for both bounding boxes
    bbox_1_min = np.minimum(bbox_1[0], bbox_1[1])
    bbox_1_max = np.maximum(bbox_1[0], bbox_1[1])
    
    bbox_2_min = np.minimum(bbox_2[0], bbox_2[1])
    bbox_2_max = np.maximum(bbox_2[0], bbox_2[1])
    
    # Check overlap in each dimension
    # For overlap to occur, the maximum of the minimums must be less than or equal to
    # the minimum of the maximums in ALL dimensions
    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)
    
    # Check if overlap exists in all dimensions
    overlap = np.all(overlap_min <= overlap_max)
    
    return overlap

#TODO: figure out the threshold
#------------------FURNITURE-FURNITURE COLLISION CHECK------------------------
# Detects furniture-furniture overlaps using bounding box vertex enclosure.
def filter_furniture_furniture_collisions(furniture_meshes, enclosure_threshold=0.4):
    collisions = []
    for i in range(len(furniture_meshes)):
        mesh_i = trimesh.Trimesh(vertices=furniture_meshes[i].vertices,
                                 faces=furniture_meshes[i].faces, process=False)
        for j in range(i + 1, len(furniture_meshes)):
            try:
                mesh_j = trimesh.Trimesh(vertices=furniture_meshes[j].vertices,
                                         faces=furniture_meshes[j].faces, process=False)

                # Check bbox overlap
                bboxi = mesh_i.bounding_box.bounds
                bboxj = mesh_j.bounding_box.bounds
                # logger.info(bboxi, bboxj)
                overlap = bbox_overlap(bboxi, bboxj)
                # logger.info("Overlap: ", overlap)
                if overlap:
                    logger.warning(f"⚠️ warning: Bounding Boxes Furniture {i}: and Furniture {j}:  overlap")
                    logger.info("So, checking vertices wise")

                    score_ij = calculate_vertex_enclosure_score(mesh_i, mesh_j)
                    score_ji = calculate_vertex_enclosure_score(mesh_j, mesh_i)
 
                    if score_ij > enclosure_threshold or score_ji > enclosure_threshold:
                        logger.warning(f"⚠️ warning: Furniture {i} overlaps with Furniture {j}, score_i_in_j={score_ij:.2f}, score_j_in_i={score_ji:.2f}")
                        collisions.append((i, j, score_ij, score_ji))
                    else:
                        logger.info("No overlap found vertices wise")

            except Exception as e:
                logger.error(f"⚠️ Error comparing Furniture {i} and Furniture {j}: {e}")
                continue

    return collisions

def point_distance(p1, p2):
    # Euclidean distance between two points
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

def triangle_area(verts):
    v0, v1, v2 = verts
    edge1 = v1 - v0
    edge2 = v2 - v0
    # Area of triangle = 0.5 * norm of cross product of two edges
    cross_prod = np.cross(edge1, edge2)
    area = 0.5 * np.linalg.norm(cross_prod)
    return area

def share_vertices(face1, face2, tolerance=1e-5):
    """
    Check if two faces share any vertices within the given tolerance.
    
    Args:
        face1: List of vertices for the first face
        face2: List of vertices for the second face
        tolerance: Distance tolerance for considering vertices as identical
        
    Returns:
        Boolean: True if faces share at least one vertex, False otherwise
    """
    for v1 in face1:
        for v2 in face2:
            if point_distance(v1, v2) < tolerance:
                return True
    return False

def is_fragmented(faces, tolerance=1e-5):
    """
    Determine if a 3D mesh is fragmented (not fully connected).
    
    Args:
        faces: List of faces, where each face is a list of vertices (points in 3D)
        tolerance: Floating point tolerance for vertex equality
    
    Returns:
        Boolean: True if the mesh is fragmented, False if fully connected
    """
    # Create a graph where vertices are faces and edges represent shared vertices
    face_graph = {}
    for i in range(len(faces)):
        face_graph[i] = set()
    
    # Build the graph by finding faces that share vertices
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            # Check if faces i and j share any vertices
            if share_vertices(faces[i], faces[j], tolerance):
                face_graph[i].add(j)
                face_graph[j].add(i)
    
    # If there are no faces, it's not fragmented
    if len(faces) == 0:
        return False
        
    # Use BFS to check if all faces are connected
    visited = set()
    queue = [0]  # Start with the first face
    visited.add(0)
    
    while queue:
        face_id = queue.pop(0)
        for neighbor in face_graph[face_id]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # If all faces are visited, the structure is not fragmented
    return len(visited) != len(faces)

def classify_face_orientation(normal, vertical_threshold=0.95, horizontal_threshold=0.95, alignment_threshold=0.99):
    """Classifies face orientation with more detailed vertical wall analysis (Z-axis is up)."""
    abs_normal = np.abs(normal)
    processed_normal = np.copy(normal)
    # Apply thresholding to normals
    if abs_normal[0] > alignment_threshold:
        processed_normal[0] = np.sign(normal[0]) * 1.0
        processed_normal[1] = 0.0
        processed_normal[2] = 0.0
    elif abs_normal[1] > alignment_threshold:
        processed_normal[1] = np.sign(normal[1]) * 1.0
        processed_normal[0] = 0.0
        processed_normal[2] = 0.0
    elif abs_normal[2] > horizontal_threshold:
        processed_normal[2] = np.sign(normal[2]) * 1.0
        processed_normal[0] = 0.0
        processed_normal[1] = 0.0
    abs_processed_normal = np.abs(processed_normal)
    if abs_processed_normal[2] > vertical_threshold:  # Primarily horizontal
        return "Horizontal Face", processed_normal
    elif abs_processed_normal[2] < 1 - vertical_threshold:  # Primarily vertical
        if abs_processed_normal[0] > alignment_threshold and abs_processed_normal[1] < 1 - alignment_threshold:
            return "Vertical Wall (Parallel to YZ Plane)", processed_normal
        elif abs_processed_normal[1] > alignment_threshold and abs_processed_normal[0] < 1 - alignment_threshold:
            return "Vertical Wall (Parallel to XZ Plane)", processed_normal
        else:
            # Analyze tilt for vertical walls
            if abs_normal[0] >= abs_normal[1]:
                return "Vertical Wall (Tilted across YZ)", processed_normal
            else:
                return "Vertical Wall (Tilted across XZ)", processed_normal
    else:
        return "do not know Face", processed_normal

# Function to calculate projected area using aligned coordinate system
def calculate_projected_area_aligned(face_indices, all_vertices, face_normals):
    if not face_indices:
        return 0.0, 0  # No faces to process
    relevant_faces = all_vertices[mesh.faces[face_indices].reshape(-1, 3)]
    relevant_normals = [face_normals[i] for i in face_indices]
    if relevant_faces.shape[0] == 0:
        return 0.0, 0
    # Calculate the average normal for this set of tilted faces
    average_normal = np.mean(relevant_normals, axis=0)
    norm = np.linalg.norm(average_normal)
    if norm > 0:
        average_normal = average_normal / norm
    else:
        logger.warning("Warning: Zero average normal for tilted faces.")
        average_normal = np.array([0.0, 1.0, 0.0]) # Default up vector
    # Create a coordinate system aligned with the average normal
    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(average_normal, [1, 0, 0]) else np.array([0, 1, 0])
    u_tilted = np.cross(average_normal, arbitrary_vector)
    norm_u_t = np.linalg.norm(u_tilted)
    if norm_u_t > 0:
        u_tilted = u_tilted / norm_u_t
    else:
        # logger.warning("Warning: Zero u_tilted for tilted faces.")
        u_tilted = np.array([1, 0, 0])
    v_tilted = np.cross(average_normal, u_tilted)
    norm_v_t = np.linalg.norm(v_tilted)
    if norm_v_t > 0:
        v_tilted = v_tilted / norm_v_t
    else:
        # logger.warning("Warning: Zero v_tilted for tilted faces.")
        v_tilted = np.array([0, 1, 0])
    # Project the vertices onto the UV plane
    origin_tilted = relevant_faces[0]
    projected_2d = np.array([[np.dot(vertex - origin_tilted, u_tilted),
                                np.dot(vertex - origin_tilted, v_tilted)]
                                for vertex in relevant_faces])
    # Calculate the bounding box of the projected vertices
    min_u = np.min(projected_2d[:, 0])
    max_u = np.max(projected_2d[:, 0])
    min_v = np.min(projected_2d[:, 1])
    max_v = np.max(projected_2d[:, 1])
    projected_area = (max_u - min_u) * (max_v - min_v)
    return projected_area, len(face_indices)

root_dataset_dir = "/mnt/sv-share/3DFRONT/data"
current_dir = "/home/ajad/Desktop/codes/LISA"

# Paths for datasets and model info
# path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
path_to_3d_front_dataset_directory = os.path.join(current_dir, "3D-FRONT-test")
path_to_model_info = os.path.join(root_dataset_dir, "3D-FUTURE-model/model_info.json")
path_to_3d_future_dataset_directory = os.path.join(root_dataset_dir, "3D-FUTURE-model")
output_dir = os.path.join(current_dir, "preprocessed-outputs")
os.makedirs(output_dir, exist_ok=True)

# Load dataset using ThreedFront class
# This loads all scenes and their associated objects, walls, etc.
d = ThreedFront.from_dataset_directory(
    path_to_3d_front_dataset_directory,
    path_to_model_info,
    path_to_3d_future_dataset_directory,
    path_to_room_masks_dir=None,
    path_to_bounds=None,
    filter_fn=lambda s: s
)

# Z-up transform matrix (to convert coordinates to Z-up convention)
transform = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

log_dir = os.path.join(current_dir, "preprocessing_logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting preprocessing...")
    

# Lists to keep track of special/invalid scenes
customized_wall_scene_ids=[]
unknown_reason=[]
count=0

# Output directories and file paths for summary and error logging
invalid_output_file_name = "invalid_scene_ids.txt"
invalid_output_path = os.path.join(output_dir, invalid_output_file_name)

unknown_output_file_name = "unknown_reason.txt"
unknown_output_path = os.path.join(output_dir, unknown_output_file_name)

# Step 1: Load invalid scene IDs (previously processed)
invalid_scene_ids_set = set()
if os.path.isfile(invalid_output_path):
    with open(invalid_output_path, "r") as f:
        for line in f:
            if "->" in line:
                scene_id = line.split("->")[0].strip()
                invalid_scene_ids_set.add(scene_id)

# print(d.scenes[0].json_path)

# Main loop: process each scene in the dataset
for s in d.scenes:  #Library-4425
    # print(s.scene_id)
    # print(s.json_path)
    # continue
    json_filename = s.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{s.scene_id}"
    scene_id = s.scene_id
    output_dir_room = os.path.join(output_dir, scene_id)

    logger.info("\n"*2)
    logger.info("="*100)
    logger.info("\n"*2)
    logger.info(f"Processing scene: {scene_id}")

    # Check if scene was already marked invalid
    if scene_unique_id in invalid_scene_ids_set:
        logger.info(f"Skipping {scene_id} as it is already marked invalid.")
        continue

    elif os.path.isdir(output_dir_room):
        logger.info(f"Skipping {scene_id} as it is valid but already processed.")
        continue

    else:
        os.makedirs(output_dir_room, exist_ok=True)
        logger.info(f"Processing {scene_id} as it is not yet processed or incomplete.")
    try:
        desired = False  # Reset for each scene
        skip_scene = False
        not_suff_wall=False
        count+=1
        extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}
        output_string =""
        door_and_window_meshes_to_concatenate =[] # to visualize windows_meshes with blue color, but here the walls are also added to it.
        wall_meshes_to_concatenate=[] # contains all meshes of walls, to check intersection with all furnitures
        furniture_meshes =[] #contains all furnitures meshes to compare intersection with walls
        wall_meshes_with_index = []
        list_of_mesh_to_concatenate = []
        wall_planes=[]
        all_relevant_objects = list(s.extras)
        non_axis_aligned_wall_count = 0

        logger.info(f"Processing Room Number: {count} ")  

        for obj_index, obj in enumerate(all_relevant_objects):
            logger.info(f"Processing object: {obj.model_type} at index {obj_index}.")
            if obj.xyz is not None and len(obj.xyz) > 0 and obj.faces is not None and len(obj.faces) > 0:
                mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
                mesh.apply_transform(transform)
                mesh_vertices = mesh.vertices
                if len(mesh_vertices) > 0:
                    z_min = mesh_vertices[:, 2].min()
                    z_max = mesh_vertices[:, 2].max()
                    height = z_max - z_min
                    is_customized_found = False #LivingRoom-35471
                    logger.info(f"Height of object {obj.model_type} at index {obj_index} is: {height}")
                else:
                    logger.warning(f"Warning: No vertices found for object at index {obj_index}. Skipping height calculation.")
                if hasattr(obj, 'model_type') and obj.model_type in ["WallOuter", "WallTop", "WallBottom", "WallInner", "Front", "Back","ExtrusionCustomizedBackgroundWall","CustomizedFeatureWall", "Floor", "Ceiling"]:
                    logger.info(f"Concatenating face: {obj.model_type}")
                    wall_meshes_to_concatenate.append(mesh)
                    wall_meshes_with_index.append((mesh, obj_index)) # Store mesh and its original index
                    list_of_mesh_to_concatenate.append(mesh) # Store mesh to concatenate
                    if hasattr(obj, 'model_type') and obj.model_type in ["WallInner", "Front", "Back"]:
                        logger.info(f"Processing as wall: {obj.model_type}")
                        # Get the broken face indices
                        face_indices = trimesh.repair.broken_faces(mesh)
                        num_faces = len(face_indices)
                        logger.info(f"Number of faces: {num_faces}")
                        logger.info(f"Face indices: {face_indices}")
                        try:
                            faces = mesh.faces[face_indices]
                            logger.info(f"Faces: {faces}")
                        except IndexError:
                            logger.error(f"[Scene {s}] Skipping scene due to: wall is not a mesh or it doesn't have broken faces")
                            logger.error("Invalid indices in 'faces'. Skipping this scene.")
                            skip_scene = True
                            with open(invalid_output_path, "a") as f:
                                f.write(f"{scene_unique_id} -> {IndexError}\n")
                            break  # break out of extras loop to skip scene

                        vertices = mesh.vertices[faces] # List of vertices of broken faces
                        flat_vertices = vertices.reshape(-1, 3)
                        logger.info(f"Vertices coordinates: \n{vertices}")
                        tolerance = 0.01


                        face_orientations = []

                        for i, face_coords in enumerate(vertices):
                            normal = np.array([0.0, 0.0, 0.0])  # Default normal
                            if len(face_coords) >= 3:
                                v1 = np.array(face_coords[0])
                                v2 = np.array(face_coords[1])
                                v3 = np.array(face_coords[2])
                                edge1 = v2 - v1
                                edge2 = v3 - v1
                                normal = np.cross(edge1, edge2)
                                norm = np.linalg.norm(normal)
                                if norm > 0:
                                    normal = normal / norm
                                else:
                                    logger.warning(f"Warning: Degenerate face at index {face_indices[i]}.")
                            else:
                                logger.warning(f"Warning: Face with fewer than 3 vertices at index {face_indices[i]}.")
                            orientation_msg, calculated_normal = classify_face_orientation(normal)
                            face_orientations.append((face_indices[i], orientation_msg, calculated_normal))

                        for index, orientation_msg, normal in face_orientations:
                            logger.info(f"Face Index: {index}, Orientation: {orientation_msg}, Normal: {normal}")
    
                        # --- Analyze the orientations of the broken faces ---
                        yz_parallel = any("Vertical Wall (Parallel to YZ Plane)" in item[1] for item in face_orientations)
                        xz_parallel = any("Vertical Wall (Parallel to XZ Plane)" in item[1] for item in face_orientations)
                        tilted_yz = any("Vertical Wall (Tilted across YZ)" in item[1] for item in face_orientations)
                        tilted_xz = any("Vertical Wall (Tilted across XZ)" in item[1] for item in face_orientations)

                        if yz_parallel and not xz_parallel and not tilted_yz and not tilted_xz:
                            identity = "parallel_to_yz_plane"
                        elif not yz_parallel and xz_parallel and not tilted_yz and not tilted_xz:
                            identity = "parallel_to_xz_plane"
                        elif yz_parallel and xz_parallel and not tilted_yz and not tilted_xz:
                            identity = "mix_parallel_planes"
                        elif not yz_parallel and not xz_parallel and tilted_yz and not tilted_xz:
                            identity = "tilted_across_yz_only"
                        elif not yz_parallel and not xz_parallel and not tilted_yz and tilted_xz:
                            identity = "tilted_across_xz_only"
                        elif (tilted_yz or tilted_xz) and (yz_parallel or xz_parallel):
                            identity = "tilted_plus_parallel"
                        elif (tilted_yz and tilted_xz and not yz_parallel and xz_parallel):
                            identity="mix_tilted_planes"#TODO: validate
                        else:
                            identity = "unknown"

                        logger.info(f"\nIdentity of face orientations:{identity}")

                        areas = np.array([triangle_area(face) for face in vertices])
                        total_broken_area = areas.sum()
                        logger.info(f"Total area of faces: {total_broken_area}")


                        faces_list = vertices.tolist()

                        # Now check if these broken faces form a fragmented mesh
                        is_frag = is_fragmented(faces_list, tolerance=tolerance)
                        logger.info(f"Are the faces fragmented? {is_frag}")
                        if is_frag:
                            logger.info("Skipping scene since fragments in the wall")
                            reason_str = "fragments in the wall"
                            with open(invalid_output_path, "a") as f:
                                f.write(f"{scene_unique_id} -> {reason_str}\n")                            
                            skip_scene = True
                            break  # Exit the extras loop; we'll check skip_scene below

                        elif not is_frag:
                            desired=True
                            
                            tolerance = 1e-6  # For comparing floating-point numbers
                            total_projected_area = 0
                            # Check for single plane cases (YZ-parallel)
                            if identity== "parallel_to_yz_plane":
                                y_min = np.min(flat_vertices[:, 1])
                                y_max = np.max(flat_vertices[:, 1])
                                z_min = np.min(flat_vertices[:, 2])
                                z_max = np.max(flat_vertices[:, 2])
                                area_yz = (y_max - y_min) * (z_max - z_min)
                                total_projected_area = area_yz
                                logger.info(f"All vertices on a plane parallel to YZ. Area: {area_yz}")
                            # Check for single plane cases (XZ-parallel)
                            elif identity== "parallel_to_xz_plane":
                                x_min = np.min(flat_vertices[:, 0])
                                x_max = np.max(flat_vertices[:, 0])
                                z_min = np.min(flat_vertices[:, 2])
                                z_max = np.max(flat_vertices[:, 2])
                                area_xz = (x_max - x_min) * (z_max - z_min)
                                total_projected_area = area_xz
                                logger.info(f"All vertices on a plane parallel  to XZ. Area: {area_xz}")
                            
                            elif identity =="mix_parallel_planes": #TODO:tilted may not be proper rectangle
                                # Handle multiple plane cases (based on constant x or y)
                                is_mix_of_parallel_planes=True
                                areas = []
                                unique_x = np.unique(np.round(flat_vertices[:, 0], 6))
                                for x_val in unique_x:
                                    on_plane = flat_vertices[np.where(np.abs(flat_vertices[:, 0] - x_val) < tolerance)]
                                    if on_plane.shape[0] >= 3:
                                        y_min = np.min(on_plane[:, 1])
                                        y_max = np.max(on_plane[:, 1])
                                        z_min = np.min(on_plane[:, 2])
                                        z_max = np.max(on_plane[:, 2])
                                        area = (y_max - y_min) * (z_max - z_min)
                                        areas.append(area)

                                unique_y = np.unique(np.round(flat_vertices[:, 1], 6))
                                for y_val in unique_y:
                                    on_plane = flat_vertices[np.where(np.abs(flat_vertices[:, 1] - y_val) < tolerance)]
                                    if on_plane.shape[0] >= 3:
                                        x_min = np.min(on_plane[:, 0])
                                        x_max = np.max(on_plane[:, 0])
                                        z_min = np.min(on_plane[:, 2])
                                        z_max = np.max(on_plane[:, 2])
                                        area = (x_max - x_min) * (z_max - z_min)
                                        areas.append(area)
                                total_projected_area = sum(areas)
                                logger.info(f"Vertices span multiple YZ or XZ parallel planes. Total projected area: {total_projected_area}")
                            
                            elif (identity=="tilted_across_yz_only" or identity=="tilted_across_xz_only") :           
                                if flat_vertices.shape[0] > 0:
                                    # 1. Determine the plane (using the normal of the first broken face)
                                    first_face_index = np.where(face_indices)[0][0]
                                    face_normal = mesh.face_normals[first_face_index]
                                    point_on_plane = flat_vertices[0]  # Take the first vertex as a point on the plane

                                    # Create a coordinate system on the plane
                                    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(face_normal, [1, 0, 0]) else np.array([0, 1, 0])
                                    u = np.cross(face_normal, arbitrary_vector)
                                    u = u / np.linalg.norm(u)
                                    v = np.cross(face_normal, u)
                                    v = v / np.linalg.norm(v)

                                    # 2. Project vertices onto the 2D plane
                                    projected_vertices_2d = np.array([[np.dot(vertex - point_on_plane, u),
                                                                    np.dot(vertex - point_on_plane, v)]
                                                                    for vertex in flat_vertices])

                                    # 3. Find the extreme points (bounding box) in 2D
                                    min_x = np.min(projected_vertices_2d[:, 0])
                                    max_x = np.max(projected_vertices_2d[:, 0])
                                    min_y = np.min(projected_vertices_2d[:, 1])
                                    max_y = np.max(projected_vertices_2d[:, 1])

                                    # 4. Calculate the area of the 2D bounding rectangle
                                    width = max_x - min_x
                                    height = max_y - min_y
                                    bounding_rectangle_area_2d = width * height
                                    logger.info(f"Area of the 2D bounding rectangle (projected): {bounding_rectangle_area_2d}")
                                else:
                                    logger.warning("No broken vertices to calculate a bounding rectangle area.")

                            elif (tilted_yz or tilted_xz) and (yz_parallel or xz_parallel):
                                identity = "tilted_plus_parallel"
                                total_projected_area = 0.0
                                
                                tilted_count = sum(1 for item in face_orientations if "Vertical Wall (Tilted across YZ)" in item[1] or "Vertical Wall (Tilted across XZ)" in item[1])
                                yz_parallel_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                                xz_parallel_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])
                                parallel_count = yz_parallel_count + xz_parallel_count
                                if tilted_count > parallel_count:
                                    dominant_orientation = "dominant_tilted"
                                    dominant_parallel_type = "n/a" # No dominant parallel type in this case
                                elif parallel_count >= tilted_count:
                                    dominant_orientation = "dominant_parallel"
                                    if yz_parallel_count >= xz_parallel_count:
                                        # Get indices of vertices belonging to YZ parallel faces
                                        yz_indices = [item[0] for item in face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1]]
                                        yz_vertices = flat_vertices[yz_indices]
                                        if yz_vertices.size > 0:  # Ensure there are vertices
                                            y_min_yz = np.min(yz_vertices[:, 1])
                                            y_max_yz = np.max(yz_vertices[:, 1])
                                            z_min_yz = np.min(yz_vertices[:, 2])
                                            z_max_yz = np.max(yz_vertices[:, 2])
                                            area_yz = (y_max_yz - y_min_yz) * (z_max_yz - z_min_yz)
                                            total_projected_area += area_yz
                                            logger.info(f"area of the yz parallel faces is: {total_projected_area}")
                                        else:
                                            logger.warning("No vertices found for YZ parallel faces.")
                                    elif xz_parallel_count > yz_parallel_count:
                                        # Get indices of vertices belonging to XZ parallel faces
                                        xz_indices = [item[0] for item in face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1]]
                                        xz_vertices = flat_vertices[xz_indices]
                                        if xz_vertices.size > 0:  # Ensure there are vertices
                                            x_min_xz = np.min(xz_vertices[:, 0])
                                            x_max_xz = np.max(xz_vertices[:, 0])
                                            z_min_xz = np.min(xz_vertices[:, 2])
                                            z_max_xz = np.max(xz_vertices[:, 2])
                                            area_xz = (x_max_xz - x_min_xz) * (z_max_xz - z_min_xz)
                                            total_projected_area += area_xz
                                            logger.info(f"area of the xz parallel faces is: {total_projected_area}")
                                        else:
                                            logger.warning("No vertices found for XZ parallel faces.")
                                parallel_faces_indices = [item[0] for item in face_orientations if "Parallel" in item[1]]
                                tilted_faces_indices = [item[0] for item in face_orientations if "Tilted" in item[1]]


                                # Calculate projected area of tilted faces (using average normal as before)
                                if tilted_faces_indices:
                                    tilted_broken_faces = mesh.faces[tilted_faces_indices]
                                    tilted_vertices = mesh.vertices[tilted_broken_faces].reshape(-1, 3)

                                    if tilted_vertices.shape[0] > 0:
                                        # 1. Calculate the average normal of the tilted faces
                                        tilted_face_normals = [item[2] for item in face_orientations if item[0] in tilted_faces_indices]
                                        if tilted_face_normals:
                                            average_normal = np.mean(tilted_face_normals, axis=0)
                                            norm = np.linalg.norm(average_normal)
                                            if norm > 0:
                                                average_normal = average_normal / norm
                                            else:
                                                logger.warning("Warning: Zero average normal for tilted faces.")
                                                average_normal = np.array([0.0, 1.0, 0.0]) # Default up vector
                                        else:
                                            logger.warning("Warning: No normals found for tilted faces.")
                                            average_normal = np.array([0.0, 1.0, 0.0]) # Default up vector

                                        # 2. Create a coordinate system aligned with the average normal
                                        arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(average_normal, [1, 0, 0]) else np.array([0, 1, 0])
                                        u_tilted = np.cross(average_normal, arbitrary_vector)
                                        norm_u_t = np.linalg.norm(u_tilted)
                                        if norm_u_t > 0:
                                            u_tilted = u_tilted / norm_u_t
                                        else:
                                            u_tilted = np.array([1, 0, 0])

                                        v_tilted = np.cross(average_normal, u_tilted)
                                        norm_v_t = np.linalg.norm(v_tilted)
                                        if norm_v_t > 0:
                                            v_tilted = v_tilted / norm_v_t
                                        else:
                                            v_tilted = np.array([0, 1, 0])

                                        # 3. Project the tilted vertices onto the UV plane
                                        origin_tilted = tilted_vertices[0]
                                        projected_tilted_2d = np.array([[np.dot(vertex - origin_tilted, u_tilted),
                                                                        np.dot(vertex - origin_tilted, v_tilted)]
                                                                        for vertex in tilted_vertices])

                                        # 4. Calculate the bounding box of the projected vertices
                                        min_u_t = np.min(projected_tilted_2d[:, 0])
                                        max_u_t = np.max(projected_tilted_2d[:, 0])
                                        min_v_t = np.min(projected_tilted_2d[:, 1])
                                        max_v_t = np.max(projected_tilted_2d[:, 1])

                                        tilted_projected_area = (max_u_t - min_u_t) * (max_v_t - min_v_t)
                                        logger.info(f"Projected area of tilted faces (aligned with average normal): {tilted_projected_area}")
                                        total_projected_area += tilted_projected_area
                                    else:
                                        logger.warning("No tilted vertices to calculate projected area.")

                                logger.info(f"Total approximate projected area (tilted + parallel): {total_projected_area}")


                            elif tilted_yz and tilted_xz and not yz_parallel and not xz_parallel:
                                identity = "mix_tilted_planes"
                                total_projected_area = 0.0

                                tilted_yz_indices = [item[0] for item in face_orientations if "Tilted across YZ" in item[1]]
                                tilted_xz_indices = [item[0] for item in face_orientations if "Tilted across XZ" in item[1]]

                                # Calculate projected area for tilted across YZ faces
                                yz_area, num_yz = calculate_projected_area_aligned(tilted_yz_indices, mesh.vertices, mesh.face_normals)
                                logger.info(f"Projected area of tilted (YZ) faces (aligned): {yz_area}")
                                total_projected_area += yz_area

                                # Calculate projected area for tilted across XZ faces
                                xz_area, num_xz = calculate_projected_area_aligned(tilted_xz_indices, mesh.vertices, mesh.face_normals)
                                logger.info(f"Projected area of tilted (XZ) faces (aligned): {xz_area}")
                                total_projected_area += xz_area

                                logger.info(f"Total approximate projected area (tilted YZ + tilted XZ - aligned): {total_projected_area}")
                        #TODO: find good threshold.
                        # i found a wall where even whena all faces were combined, it left a bit of thin vertical space.
                        # meaning we should not consider that as a gap for door and window and exclude. we need that area as threshold.
                        # or may be calculate the area of normal door and keep a value lower than that.
                        # it seems doors area is usually above 1.
                        area_tolerance=0.9
                        if total_projected_area>total_broken_area+area_tolerance and not is_frag: # add either conditions like if faces_broken=2 or less.
                            #print("there is enough space available for either a wall or a window.")
                            #print("--------------------------------------------------")
                            #print("now, find where to fit a door." \
                            # "for the parallel on any on of the axis, its already working." \
                            # "for the one with mixture of parallel faces, find the largest one, and perform as above." \
                            # "for the slaned ones... think")
                            logger.info(f"There is enough space available for either a wall or a window.")
                        # Case1: in any of the plane parallel.
                            if identity=="parallel_to_yz_plane":
                                constant_coord = 'x'
                                constant_value = np.mean(flat_vertices[:, 0])  # take mean
                                remaining_coords = flat_vertices[:, 1:3]  # y and z
                                logger.info(f"parallel_to_yz_plane")
                                
                            elif identity=="parallel_to_xz_plane":
                                constant_coord = 'y'
                                constant_value = np.mean(flat_vertices[:, 1])  # take mean
                                remaining_coords = flat_vertices[:, [0, 2]] 
                                logger.info(f"parallel_to_xz_plane")
                                
                            #we prioritize the plane with maximum number of vertices, likely only that has the door/window
                            #TODO:#but can be refined.
                            elif identity == "mix_parallel_planes":
                                yz_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                                xz_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])

                                if yz_count >= xz_count:
                                    dominant_orientation = "dominant_yz_parallel"
                                    constant_coord = 'x'
                                    constant_value = np.mean(flat_vertices[:, 0])  # take mean
                                    remaining_coords = flat_vertices[:, 1:3]  # y and z
                                elif xz_count > yz_count:
                                    constant_coord = 'y'
                                    constant_value = np.mean(flat_vertices[:, 1])  # take mean
                                    remaining_coords = flat_vertices[:, [0, 2]] 
                                    dominant_orientation = "dominant_xz_parallel"
                                # else:
                                #     dominant_orientation = "equal_yz_xz_parallel" # Or handle this case as needed
                                logger.info(f"Mixed but prioritiezed parallel_to_xz_plane")
                                logger.info(f"YZ Parallel Count: {yz_count}, XZ Parallel Count: {xz_count}, Dominant: {dominant_orientation}")

                            #Case 2: tilted and parallel combination
                            #we prioritize the parallel one and assume it contains the doors/windows.
                            elif identity == "tilted_plus_parallel":
                                tilted_count = sum(1 for item in face_orientations if "Vertical Wall (Tilted across YZ)" in item[1] or "Vertical Wall (Tilted across XZ)" in item[1])
                                yz_parallel_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                                xz_parallel_count = sum(1 for item in face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])
                                parallel_count = yz_parallel_count + xz_parallel_count
                                if tilted_count > parallel_count:
                                    dominant_orientation = "dominant_tilted"
                                    dominant_parallel_type = "n/a"
                                elif parallel_count >= tilted_count:
                                    dominant_orientation = "dominant_parallel"
                                    if yz_parallel_count > xz_parallel_count:
                                        dominant_parallel_type = "dominant_yz_parallel"
                                        constant_coord = 'x'
                                        constant_value = np.mean(flat_vertices[:, 0])  # take mean
                                        remaining_coords = flat_vertices[:, 1:3]  # y and z

                                    elif xz_parallel_count > yz_parallel_count:
                                        dominant_parallel_type = "dominant_xz_parallel"
                                        constant_coord = 'y'
                                        constant_value = np.mean(flat_vertices[:, 1])  # take mean
                                        remaining_coords = flat_vertices[:, [0, 2]] 
                                        dominant_orientation = "dominant_xz_parallel"
                                        
                                logger.info(f"Tilted Count: {tilted_count}, YZ Parallel Count: {yz_parallel_count}, XZ Parallel Count: {xz_parallel_count}, Dominant Category: {dominant_orientation}, Dominant Parallel Type: {dominant_parallel_type}")
                    
                                    
                            # Step 3: Extract the unique values for the remaining coordinates (y and z or x and z or x and y)
                            unique_remaining_coords = np.unique(remaining_coords, axis=0)

                            # Step 4: Get the min and max for the remaining coordinates
                            min_coord = np.min(unique_remaining_coords, axis=0)
                            max_coord = np.max(unique_remaining_coords, axis=0)

                            logger.info(f"Constant coordinate: {constant_coord} = {constant_value}")
                            # Step 5: Identify the missing gap (door) in the remaining coordinates
                            # We'll check for gaps in the sorted unique coordinates
                            # Given a wall, there is a section void ( so we check the range in which it is void)
                            sorted_remaining_coords = np.sort(unique_remaining_coords, axis=0)
                            logger.info(f"sorted: {sorted_remaining_coords}")

                            #Remove duplicacy
                            sorted_remaining_coords = np.unique(sorted_remaining_coords, axis=0)
                            logger.info(f"Unique sorted remaining coords:\n{sorted_remaining_coords}")
                            first_column = sorted_remaining_coords[:, 0]
                            logger.info(f"First column of remaining coords:\n{first_column}")
                            mean_first_column = np.mean(first_column)
                            logger.info(f"Mean of the first column: {mean_first_column}")

                            # Initialize min_1 and max_1 as None
                            min_1 = None
                            max_1 = None

                            distances_from_mean = np.abs(first_column - mean_first_column)
                            logger.info(f"Distances from mean: {distances_from_mean}")
                            # Step 2: Sort the points by how close they are to the mean and get their original indices
                            sorted_indices = np.argsort(distances_from_mean)
                            sorted_points = first_column[sorted_indices]
                            logger.info(f"Sorted points: {sorted_points}")
                            # Step 3: Get the indices of the two points closest to the mean
                            nearest_two_indices = sorted_indices[:2]
                            nearest_two_points = first_column[nearest_two_indices]
                            logger.info(f"Two nearest points to the mean: {nearest_two_points}")

                            if len(nearest_two_points) == 2:  # Ensure we have at least two points
                                distance_nearest = np.abs(nearest_two_points[0] - nearest_two_points[1])
                                logger.info(f"Distance between the two nearest points: {distance_nearest}")
                                if distance_nearest > 0.5:
                                    logger.info(f"Found suitable nearest points: {nearest_two_points[0]} and {nearest_two_points[1]}")
                                    min_1 = np.min(nearest_two_points)
                                    max_1 = np.max(nearest_two_points)
                                    logger.info(f"min_1 = {min_1}, max_1 = {max_1}")
                                else:
                                    logger.info("The two nearest points are too close. Searching for other suitable pairs...")
                                    found_suitable = False
                                    for i in range(len(sorted_points) - 1):
                                        for j in range(i + 1, len(sorted_points)):
                                            point1 = sorted_points[i]
                                            point2 = sorted_points[j]
                                            distance_between_points = np.abs(point1 - point2)
                                            logger.info(f"Checking distance between {point1} and {point2}: {distance_between_points}")
                                            if distance_between_points > 0.5:
                                                logger.info(f"Found suitable points: {point1} and {point2}")
                                                logger.info(f"Distance between them: {distance_between_points}")
                                                min_1 = np.min([point1, point2])
                                                max_1 = np.max([point1, point2])
                                                logger.info(f"min_1 = {min_1}, max_1 = {max_1}")
                                                found_suitable = True
                                                break  # Exit the inner loop once a suitable pair is found
                                        if found_suitable:
                                            break  # Exit the outer loop if a suitable pair is found

                                    if not found_suitable:
                                        logger.info("No suitable pair of points found with distance > 0.5")
                            else:
                                logger.info("Not enough points to compare.")

                            if len(sorted_remaining_coords[:,0])<4:
                                min_1=None
                                max_1=None
                                logger.info(f"min_1 = {min_1}, max_1 = {max_1}")
                                logger.info("the wall faces is such that, there should be no window... only 3 points on one axis")

                            logger.info(f"Adjusted min_1 (less than mean): {min_1}")
                            logger.info(f"Adjusted max_1 (greater than mean): {max_1}")

                            # Assuming you're interested in the z-axis (second column)
                            z_coordinates = sorted_remaining_coords[:, 1]
                            sorted_z = np.sort(z_coordinates)
                            differences = np.diff(sorted_z)

                            if differences.size > 0:
                                largest_diff_index = np.argmax(differences)
                                largest_difference = differences[largest_diff_index]

                                val1 = sorted_z[largest_diff_index]
                                val2 = sorted_z[largest_diff_index + 1]

                                if val1 < val2:
                                    min2 = val1
                                    max2 = val2
                                else:
                                    min2 = val2
                                    max2 = val1

                                logger.info(f"min z value: {min2}")
                                logger.info(f"max z value: {max2}")
                            else:
                                logger.info("Not enough unique points to calculate consecutive differences.") 
                            min_2 = min2
                            max_2 = max2
                            logger.info(f"min_2 = {min_2}, max_2 = {max_2}")

                            # Step 4: Print the results for the gaps in the remaining axes
                            if min_1 is not None and max_1 is not None:
                                logger.info(f" gap found in the first remaining coordinate: from {min_1} to {max_1}") 
                            else:
                                logger.info("No  gap found in the first remaining coordinate.")

                            if min_2 is not None and max_2 is not None:
                                logger.info(f" gap found in the second remaining coordinate: from {min_2} to {max_2}")
                            else:
                                logger.info("No  gap found in the second remaining coordinate.")
                                    
                            is_door = False

                            if min_2 is not None and min_1 is not None and max_2 is not None and max_1 is not None:
                                if min_2 <= 0.1:  
                                    is_door = True
                                    logger.info(f"This is a Door, starting from the floor (z_min).")
                                else:
                                    is_window=True
                                    logger.info(f"This is a Window.")
                                if constant_coord == 'x':
                                            gap_thickness = 0.05  # Example thickness for the gap
                                            logger.info(f"gap_thickness: {gap_thickness}")
                                            width = max_1 - min_1
                                            height= max_2 - min_2
                                            logger.info(f"width:{width} and height:{height}")
                                            gap_vertices = np.array([
                                                [constant_value, min_1, min_2],
                                                [constant_value, max_1, min_2],
                                                [constant_value, max_1, max_2],  # Assuming z_max is defined
                                                [constant_value, min_1, max_2],  # Assuming z_max is defined
                                                [constant_value + gap_thickness, min_1, min_2],
                                                [constant_value + gap_thickness, max_1, min_2],
                                                [constant_value + gap_thickness, max_1, max_2],
                                                [constant_value + gap_thickness, min_1, max_2]
                                            ])
                                            gap_faces = np.array([
                                                [0, 1, 2], [0, 2, 3],   # Front face
                                                [4, 5, 6], [4, 6, 7],   # Back face
                                                [0, 4, 7], [0, 7, 3],   # Bottom face
                                                [1, 5, 6], [1, 6, 2],   # Top face
                                                [0, 1, 5], [0, 5, 4],   # Left face
                                                [3, 2, 6], [3, 6, 7]    # Right face
                                            ])
                                            logger.info(f"gap_vertices: {gap_vertices}")
                                else:  # constant_coord is 'y' (assuming this is the only other possibility)
                                            gap_thickness = 0.05  # Example thickness for the gap
                                            width = max_1 - min_1
                                            height =max_2 - min_2
                                            logger.info(f"width:{width} and height:{height}")
                                            gap_vertices = np.array([
                                                [min_1, constant_value, min_2],
                                                [max_1, constant_value, min_2],
                                                [max_1, constant_value, max_2],  # Assuming z_max is defined
                                                [min_1, constant_value, max_2],  # Assuming z_max is defined
                                                [min_1, constant_value + gap_thickness, min_2],
                                                [max_1, constant_value + gap_thickness, min_2],
                                                [max_1, constant_value + gap_thickness, max_2],
                                                [min_1, constant_value + gap_thickness, max_2]
                                            ])
                                            # Just indexing the vertices to make faces for visualization
                                            gap_faces = np.array([
                                                [0, 1, 2], [0, 2, 3],   # Front face
                                                [4, 5, 6], [4, 6, 7],   # Back face
                                                [0, 4, 7], [0, 7, 3],   # Bottom face
                                                [1, 5, 6], [1, 6, 2],   # Top face
                                                [0, 4, 7], [0, 7, 3],   # Left face
                                                [3, 2, 6], [3, 6, 7]    # Right face
                                            ])
                                            logger.info(f"gap_faces: {gap_faces}")
                                door_window_mesh = trimesh.Trimesh(vertices=gap_vertices, faces=gap_faces)
                                    
                                # Creating a simple mesh for door/window from the gap vertices (assuming quad face)
                                logger.info(f"Shape of gap_vertices: {gap_vertices.shape}")
                                logger.info(f"Content of gap_vertices:\n {gap_vertices}")
                                #TODO: validate this: since it will save computation if we don't need to calculate intersection

                                #best_wall_id, iou_score = find_closest_wall_by_vertex_enclosure(obj, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
                                iou_score=0.5#just for fun, discard it.
                                best_wall_id = obj.model_uid
                                logger.info(f"Wall ID: {obj.model_uid}")

                                if best_wall_id is not None and iou_score > 0.0: # Only process if a wall with sufficient IoU is found
                                    wall_id = extra2index.get(best_wall_id, 0)
                                    logger.info(f"wall_id: {wall_id}")
                                    front_face_indices = np.array([0, 1, 2, 3])
                                    front_face_vertices = door_window_mesh.vertices[front_face_indices]
                                    position = front_face_vertices.mean(axis=0)
                                    position_x = position[0]
                                    position_y = position[1]
                                    position_z = position[2]
                                    logger.info(f"position_x: {position_x}, position_y: {position_y}, position_z: {position_z}")

                                    # Add to the appropriate list (window or door)
                                    if is_door:
                                        door_and_window_meshes_to_concatenate.append(door_window_mesh)
                                        # list_of_mesh_to_concatenate.append(door_window_mesh)
                                        logger.info(f"Wall id for door (by IoU) is : {wall_id}, IoU: {iou_score}")
                                        door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
                                        logger.info(f"door_obj: {door_obj.to_language_string()}")
                                        output_string += door_obj.to_language_string() + '\n'
                                    else: #windows
                                        door_and_window_meshes_to_concatenate.append(door_window_mesh)
                                        # list_of_mesh_to_concatenate.append(door_window_mesh)
                                        logger.info(f"Wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
                                        win_obj = Window(obj_index, wall_id, position_x, position_y, position_z, width, height, "window")
                                        logger.info(f"win_obj: {win_obj.to_language_string()}")
                                        output_string += win_obj.to_language_string() + '\n' 
                                else:
                                    logger.info("No wall matched to place door or window...")
                            else:
                                logger.info("This wall also doesnt have a gap........................")
                        else:
                            logger.info("This is just a wall and no door/window in it.")

                        floor_vertices = mesh_vertices[np.isclose(mesh_vertices[:, 2], z_min, atol=1e-3)]
                        floor_vertices = np.unique(floor_vertices, axis=0)

                        # print(floor_vertices)

                        if len(floor_vertices) < 2:
                            logger.info("Skipping malformed wall...")
                            continue  # skip malformed wall

                        xz = floor_vertices[:, [0, 1]]  # X-Y plane since Z is up
                        dists = squareform(pdist(xz))
                        i0, i1 = np.unravel_index(np.argmax(dists), dists.shape)

                        start = floor_vertices[i0]
                        end = floor_vertices[i1]
                        ax, ay, az = start[0], start[1], start[2]
                        bx, by, bz = end[0], end[1], end[2]

                        thickness = mesh.bounding_box.primitive.extents.min()
                        thickness = max(thickness, 0.08)
                        mesh_vertices = mesh.vertices
                        z_min = mesh_vertices[:, 2].min()
                        z_max = mesh_vertices[:, 2].max()
                        height = z_max - z_min

                        # Use a consistent index for Wall objects, even if they come from the combined list
                        wall = Wall(obj_index, ax, ay, az, bx, by, bz, height, thickness)
                        output_string += wall.to_language_string() + '\n'
                        logger.info(f"wall: {wall.to_language_string()}")


                    elif hasattr(obj, 'model_type') and obj.model_type in ["ExtrusionCustomizedBackgroundWall","CustomizedFeatureWall"]:
                        reason_str="customized wall"
                        logger.info(f"Customized wall added to invalid scene ids.")
                        with open(invalid_output_path, "a") as f:
                            f.write(f"{scene_unique_id} -> {reason_str}\n")
                        logger.info(f"Skipping this scene due to customized wall.")

                        skip_scene = True
                        break  # Exit the extras loop
                else:
                    logger.info(f"Skipping language generation for {obj.model_type}(This is not wall.)")
            else:
                logger.info(f"Warning: Skipping object {obj.model_type} at index {obj_index} since invalid or empty xyz or faces data.")
   

        if len(wall_meshes_to_concatenate) <2:
            reason_str =f"no sufficient walls, only {len(wall_meshes_to_concatenate)} walls"
            not_suff_wall=True
            logger.info(f"Not Enough Walls: {not_suff_wall}")
            logger.info(f"Reason: {reason_str}")
            # Record invalid scene once
            with open(invalid_output_path, "a") as f:
                f.write(f"{scene_unique_id} -> {reason_str}\n")

              
        if skip_scene:
            logger.info("Skipping rest of the scene including furniture processing since skip_scene is True.")
            continue  # Skip rest of the scene including furniture processing

        if not_suff_wall:
            logger.info("Skipping rest of the scene including furniture processing since Not Enough Walls.")
            continue 
            

        for index, furniture in enumerate(s.bboxes):
            try:
                # 1. Load raw mesh
                mesh = furniture.raw_model()
                class_name = furniture.label

                logger.info(f"class_name: {class_name}")

                if class_name == "unknown_category":
                    logger.warning("Skipping furniture with unknown category")
                    continue

                # 2. Apply model-to-scene transform
                furniture_vertices = furniture._transform(mesh.vertices)
                mesh = trimesh.Trimesh(vertices=furniture_vertices, faces=mesh.faces)

                # 3. Apply Z-up transformation (if needed)
                mesh.apply_transform(transform)
                furniture_meshes.append(mesh)
                list_of_mesh_to_concatenate.append(mesh)

                # 4. Get bounding box
                bbox = mesh.bounding_box
                center = bbox.primitive.transform[:3, 3]
                dims = bbox.extents

                rot_matrix = bbox.primitive.transform[:3, :3]
                heading = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # yaw in radians

                bbox_obj = Bbox(index, class_name, center[0], center[1], center[2],
                                heading, dims[0], dims[1], dims[2])

                logger.info(f"bbox_obj: {bbox_obj.to_language_string()}")
                output_string += bbox_obj.to_language_string() + '\n'

            except Exception as e:
                logger.error(f"[Error] Index {index} - {class_name}: {e}")

                with open(invalid_output_path, "a") as f:
                    f.write(f"{scene_unique_id} -> {e}\n")
                continue


        if desired:
            #Perform furniture-furniture and furniture-wall collision testing...
            valid_furniture, invalid_ids = filter_furniture_in_scene(wall_meshes_to_concatenate, furniture_meshes, enclosure_threshold=0.7)
            if not invalid_ids:
                logger.info("✅ All furniture items are considered valid (enclosed by the walls).")
            else:
                logger.info("Not valid furniture items.")
                reason_str =f"wall-furniture collision..."
                with open(invalid_output_path, "a") as f:
                    f.write(f"{scene_unique_id} -> {reason_str}\n")                            
                logger.info(f"Reason: {reason_str}")

            furniture_furniture_collisions = filter_furniture_furniture_collisions(furniture_meshes)
            if furniture_furniture_collisions:
                reason_str =f"furniture-furniture collision..."
                with open(invalid_output_path, "a") as f:
                    f.write(f"{scene_unique_id} -> {reason_str}\n")                            
            else:
                logger.info("✅ No undesired furniture-furniture collisions.")

            #Proceed if no colliions only
            if not furniture_furniture_collisions and not invalid_ids:
                logger.info("Operation before saving the script")
                lines = output_string.strip().split('\n')
                def sort_key(line):
                    if line.startswith("wall"):
                        return 0, line
                    elif line.startswith("window"):
                        return 1, line
                    elif line.startswith("door"):
                        return 2, line
                    elif line.startswith("bbox"):
                        return 3, line
                    else:
                        return 4, line

                sorted_lines = sorted(lines, key=sort_key)
                formatted_output = "\n".join(sorted_lines)
                logger.info("Formatted in the chronological order of wall, door, window, bbox...")

                # -------------Saving the meshes for visualization ------------------
                # Only layout represent the scene based on script. other are only for understandiing and validation 
                # To visualize furniture meshes to visualize: yellow color
                if list_of_mesh_to_concatenate:
                    concatenated_mesh = trimesh.util.concatenate(list_of_mesh_to_concatenate)
                    output_filename = os.path.join(output_dir_room, "concatenated_mesh.obj")
                    concatenated_mesh.export(output_filename)
                    logger.info(f"Succesfully concatenated and exported to {output_filename}")
                else:
                    logger.info("No valid furnitures meshes to concatenate")
                #To visualize windows and door meshes : sky blue color
                if door_and_window_meshes_to_concatenate:
                    concatenated_windows_mesh = trimesh.util.concatenate(door_and_window_meshes_to_concatenate)
                    output_filename = os.path.join(output_dir_room, "concatenated_windows_mesh.obj")
                    concatenated_windows_mesh.export(output_filename)
                    logger.info(f"Succesfully concatenated windows and doors and exported to {output_filename}")
                else:
                    logger.info("No valid windows meshes to concatenate")
                #To visualize wall meshes : brown color
                if wall_meshes_to_concatenate:
                    concatenated_walls_mesh = trimesh.util.concatenate(wall_meshes_to_concatenate)
                    output_filename = os.path.join(output_dir_room, "concatenated_walls_mesh.obj")
                    concatenated_walls_mesh.export(output_filename)
                    logger.info(f"succesffuly concatenated walls and exported to {output_filename}")
                else:
                    logger.info("No valid walls meshes to concatenate")
                #To visualize furniture meshes : yellow color
                if furniture_meshes:
                    concatenated_furniture_mesh = trimesh.util.concatenate(furniture_meshes)
                    output_filename = os.path.join(output_dir_room, "concatenated_furniture_mesh.obj")
                    concatenated_furniture_mesh.export(output_filename)
                    logger.info(f"succesffuly concatenated furniture and exported to {output_filename}")
                else:
                    logger.info("No valid furniture meshes to concatenate")

                #Change the script: bbox1,bbox2,.... ===>bobx and add name.
                #Input: formatted_output_cleaned 
      
                visualize_formatted_output_cleaned=formatted_output
                formatted_output_cleaned=re.sub(r'\bbbox_\d+=', 'bbox=', formatted_output)

                # Extract room name from scene_unique_id
                match = re.search(r'__(\w+)-', scene_unique_id)
                room_name = match.group(1) if match else "Unknown"
                formatted_output_cleaned = f"room=Room({room_name})\n" + formatted_output_cleaned.strip()

                #----------------SAVING the script -----------------------------
                output_file_name= f"{scene_unique_id}.txt"
                script_output_path = os.path.join(output_dir_room, output_file_name)
                with open(script_output_path,"w")as f:
                    f.write(formatted_output_cleaned)
                logger.info(f"✅ Succesfully wrote script to :{script_output_path}")
                output_file_name= f"{scene_unique_id}_vis.txt"
                visualize_script_output_path = os.path.join(output_dir_room, output_file_name)
                with open(visualize_script_output_path,"w")as f:
                    f.write(visualize_formatted_output_cleaned)
                logger.info(f"✅ Succesfully wrote script to :{visualize_script_output_path}")
                valid_output_file_name= f"valid_scene_ids.txt"
                valid_output_path = os.path.join(output_dir, valid_output_file_name)
                with open(valid_output_path, "a") as f:
                    f.write(f"{scene_unique_id}\n")
                logger.info("Added to summary...")
                logger.info("Completed Processing scene:", s.scene_id)
            else:
                logger.info("❌ not the desire scene. marking as invalid")
                logger.info("Completed Processing scene:", s.scene_id)

    except Exception as e:
        json_filename = s.json_path.split("/")[-1].replace(".json", "")
        scene_unique_id = f"{json_filename}__{s.scene_id}"
        logger.error(f"Failed: {scene_unique_id} — {e}")
        unknown_file_name = "unknown_reason.txt"
        invalid_output_path = os.path.join(output_dir, unknown_file_name)
        with open(invalid_output_path, "a") as f:
            f.write(f"{scene_unique_id} -> {e}\n")