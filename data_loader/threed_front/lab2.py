import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from itertools import combinations
import pdb  # Import the debugger
import warnings
import gc
from scipy.spatial import ConvexHull
import sys




#TODO:

# 1. for the scenes whose wall is not aligned to the axis.
#2. find better estimate for furniture and wall intersection.
#since, objects can be large and cross the wall, and not many vertices may fall on the wall. it is a problem.
#3. condition of wthere a wall only or contaisn window and door



#-----------FURNITURE AND WALL COLLISION DETECTION-------------
def is_point_in_bbox(point, bbox_min, bbox_max, tolerance=1e-6):
    """Checks if a point is within a bounding box with a small tolerance."""
    return all(bbox_min[i] - tolerance <= point[i] <= bbox_max[i] + tolerance for i in range(3))

def calculate_vertex_enclosure_score(mesh1, mesh2):
    """Calculates the percentage of vertices of mesh1 inside the bounding box of mesh2."""
    bbox_min2, bbox_max2 = mesh2.bounding_box.bounds
    enclosed_count = sum(is_point_in_bbox(vertex, bbox_min2, bbox_max2) for vertex in mesh1.vertices)
    return enclosed_count / len(mesh1.vertices) if mesh1.vertices.size > 0 else 0










# #TODO: figure out the threshold
# def filter_furniture_in_scene(wall_meshes, furniture_meshes, enclosure_threshold=0.5):
#     # Combine wall meshes into one large mesh (to save memory)
#     combined_vertices = []
#     combined_faces = []
#     vertex_offset = 0

#     for wall_mesh in wall_meshes:
#         combined_vertices.append(wall_mesh.vertices)
#         combined_faces.append(wall_mesh.faces + vertex_offset)
#         vertex_offset += len(wall_mesh.vertices)

#     if not combined_vertices:
#         return []  # No walls, exclude all furniture

#     # Create the combined wall mesh
#     combined_wall_mesh = trimesh.Trimesh(
#         vertices=np.vstack(combined_vertices),
#         faces=np.vstack(combined_faces),
#         process=False
#     )

#     # Now check each furniture against the combined wall mesh
#     valid_furniture = []
#     for i, furniture_mesh in enumerate(furniture_meshes):
#         try:
#             # Create the furniture mesh object
#             mesh = trimesh.Trimesh(vertices=furniture_mesh.vertices, faces=furniture_mesh.faces, process=False)

#             # Calculate how well the furniture mesh is enclosed by the combined wall mesh
#             score = calculate_vertex_enclosure_score(mesh, combined_wall_mesh)

#             # If the score is above the threshold, consider it valid
#             if score >= enclosure_threshold:
#                 valid_furniture.append(furniture_mesh)
#             else:
#                 # Print the collision details if the furniture doesn't meet the threshold
#                 print(f"⚠️ Furniture {i} collides with walls, enclosure score: {score:.2f}")

#         except Exception as e:
#             print(f"⚠️ Error processing furniture mesh {i}: {e}")
#             continue

#     # Clean up memory after the check
#     del combined_wall_mesh
#     gc.collect()  # Trigger garbage collection to free memory

#     return valid_furniture







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
                print(f"⚠️ warning: Furniture {i} collides with walls, enclosure score: {score:.2f}")
        except Exception as e:
            print(f"⚠️ Error processing furniture mesh {i}: {e}")
            logger.error(f"⚠️ Error processing furniture mesh {i}: {e}")
            invalid_indices.append(i)
            continue

    del combined_wall_mesh
    gc.collect()

    return valid_furniture, invalid_indices



#TODO: figure out the threshold
#------------------FURNITURE-FURNITURE COLLISION CHECK------------------------
def filter_furniture_furniture_collisions(furniture_meshes, enclosure_threshold=0.4):
    """Detects furniture-furniture overlaps using bounding box vertex enclosure."""
    collisions = []
    for i in range(len(furniture_meshes)):
        mesh_i = trimesh.Trimesh(vertices=furniture_meshes[i].vertices,
                                 faces=furniture_meshes[i].faces, process=False)
        for j in range(i + 1, len(furniture_meshes)):
            try:
                mesh_j = trimesh.Trimesh(vertices=furniture_meshes[j].vertices,
                                         faces=furniture_meshes[j].faces, process=False)

                # Check if i's vertices are inside j's bounding box
                score_ij = calculate_vertex_enclosure_score(mesh_i, mesh_j)
                score_ji = calculate_vertex_enclosure_score(mesh_j, mesh_i)
                avg_score = (score_ij + score_ji) / 2.0

                if score_ij > enclosure_threshold or score_ji > enclosure_threshold:
                    print(f"⚠️ warning: Furniture {i} overlaps with Furniture {j}, score_i_in_j={score_ij:.2f}, score_j_in_i={score_ji:.2f}")
                    collisions.append((i, j, score_ij, score_ji))

            except Exception as e:
                print(f"⚠️ Error comparing Furniture {i} and Furniture {j}: {e}")
                continue

    return collisions









def log(*args):
    print("=" * 10, *args, "=" * 10)





# Paths
#TODO: uncomment
path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
# path_to_3d_front_dataset_directory = "/home/ajad/Desktop/codes/LISA"
path_to_model_info = "/mnt/sv-share/3DFRONT/3D-FUTURE-model/model_info.json"
path_to_3d_future_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FUTURE-model"

# Load dataset
d = ThreedFront.from_dataset_directory(
    path_to_3d_front_dataset_directory,
    path_to_model_info,
    path_to_3d_future_dataset_directory,
    path_to_room_masks_dir=None,
    path_to_bounds=None,
    filter_fn=lambda s: s
)

# Z-up transform
transform = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


invalid_scene_ids = []
valid_scene_ids=[]
customized_wall_scene_ids=[]

count=0


for s in d.scenes:  #Library-4425
    desired = False  # Reset for each scene
    skip_scene = False
    not_suff_wall=False
    json_filename = s.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{s.scene_id}"
    # print(f"scene  id is: {scene_unique_id}")

    # if s.scene_id in [ "Bedroom-515", "LivingDiningRoom-8375", "Bedroom-15517", "LivingDiningRoom-10021", "Bedroom-5885", "Bedroom-10062", "Bedroom-10078", "Library-4425","LivingDiningRoom-3474", "Hallway-1213", "SecondBedroom-7177" ] :
    #     continue  #MasterBedroom-5863 "MasterBedroom-5863", "Bedroom-515",

    # if s.scene_id not in ["LivingRoom-35471","LivingRoom-34996 ","Bedroom-34900","DiningRoom-35093"] : #LivingDiningRoom-3474
    #     continue  #MasterBedroom-5863


    log(f"Count: {count} ")  
    log(f"Processing scene: {s.scene_id} ")    
    log(f"Processing scene: {json_filename} ")  
    count+=1
    print("")
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}

    output_string =""
   
    windows_meshes_to_concatenate =[] # to visualize windows_meshes with blue color, but here the walls are also added to it.
    wall_meshes_to_concatenate=[] # contains all meshes of walls, to check intersection with all furnitures
    furniture_meshes =[] #contains all furnitures meshes to compare intersection with walls

    wall_meshes_with_index = []

    wall_planes=[]


    all_relevant_objects = list(s.extras)

    non_axis_aligned_wall_count = 0
    for obj_index, obj in enumerate(all_relevant_objects):
        if obj.xyz is not None and len(obj.xyz) > 0 and obj.faces is not None and len(obj.faces) > 0:
            mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
            mesh.apply_transform(transform)

            vertices = mesh.vertices
            if len(vertices) > 0:
                z_min = vertices[:, 2].min()
                z_max = vertices[:, 2].max()
                height = z_max - z_min
                is_customized_found = False #LivingRoom-35471
            else:
                print(f"Warning: No vertices found for object at index {obj_index}. Skipping height calculation.")


            #"CustomizedFeatureWall",""WallOuter",  "ExtrusionCustomizedBackgroundWall",    
            if hasattr(obj, 'model_type') and obj.model_type in [ "WallInner", "Front", "Back","ExtrusionCustomizedBackgroundWall","CustomizedFeatureWall"]:
                wall_meshes_to_concatenate.append(mesh)
                wall_meshes_with_index.append((mesh, obj_index)) # Store mesh and its original index
                # if hasattr(obj, 'model_type') and obj.model_type in ["CustomizedFeatureWall"]:

                #     print("what to do!")

                
                if hasattr(obj, 'model_type') and obj.model_type in ["WallInner", "Front", "Back"]:
                    

                    # # Get the broken face indices
                    # broken = trimesh.repair.broken_faces(mesh)
                    # num_broken = len(broken)
                    # print(f"num_broken_faces:{num_broken}")
                    # broken_faces = mesh.faces[broken]
                    # broken_vertices = mesh.vertices[broken_faces]
                    # flat_vertices = broken_vertices.reshape(-1, 3)
                    # print(f"Broken face indices: {broken}")
                    # print(f"Broken vertices coordinates: \n{broken_vertices}")
                    # tolerance = 0.01



                # handles oriented and axis aligned walls.

                    import numpy as np
                    import trimesh
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

                    print("")
                    print("")
                    print("")
                    print("---------------------------------------")
                    # Get the broken face indices
                    broken = trimesh.repair.broken_faces(mesh)
                    num_broken = len(broken)
                    print(f"num_broken_faces:{num_broken}")


                    try:
                        broken_faces = mesh.faces[broken]
                    except IndexError:
                        print(f"[Scene {s}] Skipping scene due to: wall is not a mesh or it doesn't have broken faces")
                        invalid_scene_ids.append((scene_unique_id, IndexError))
                        warnings.warn("Invalid indices in 'broken'. Skipping this scene.")
                        logger.error("Invalid indices in 'broken'. Skipping this scene.")
                        skip_scene = True
                        break  # break out of extras loop to skip scene



                    broken_vertices = mesh.vertices[broken_faces]
                    flat_vertices = broken_vertices.reshape(-1, 3)
                    print(f"Broken face indices: {broken}")
                    print(f"Broken vertices coordinates: \n{broken_vertices}")
                    tolerance = 0.01









                    broken_face_orientations = []

                    for i, face_coords in enumerate(broken_vertices):
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
                                print(f"Warning: Degenerate broken face at index {broken[i]}.")
                        else:
                            print(f"Warning: Broken face with fewer than 3 vertices at index {broken[i]}.")

                        orientation_msg, calculated_normal = classify_face_orientation(normal)
                        broken_face_orientations.append((broken[i], orientation_msg, calculated_normal))

                    print("\nOrientations of the broken faces:")
                    for index, orientation_msg, normal in broken_face_orientations:
                        print(f"Broken Face Index: {index}, Orientation: {orientation_msg}, Normal: {normal}")




                                    # --- Analyze the orientations of the broken faces ---
                    yz_parallel = any("Vertical Wall (Parallel to YZ Plane)" in item[1] for item in broken_face_orientations)
                    xz_parallel = any("Vertical Wall (Parallel to XZ Plane)" in item[1] for item in broken_face_orientations)
                    tilted_yz = any("Vertical Wall (Tilted across YZ)" in item[1] for item in broken_face_orientations)
                    tilted_xz = any("Vertical Wall (Tilted across XZ)" in item[1] for item in broken_face_orientations)

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

                    print("\nIdentity of broken face orientations:", identity)
                        

                    def triangle_area(verts):
                        v0, v1, v2 = verts
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        # Area of triangle = 0.5 * norm of cross product of two edges
                        cross_prod = np.cross(edge1, edge2)
                        area = 0.5 * np.linalg.norm(cross_prod)
                        return area

                    areas = np.array([triangle_area(face) for face in broken_vertices])
                    total_broken_area = areas.sum()
                    print(f"Total area of broken faces: {total_broken_area}")









    

                                                
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

                    def point_distance(p1, p2):
                        """
                        Calculate Euclidean distance between two 3D points.
                        
                        Args:
                            p1: First point [x, y, z]
                            p2: Second point [x, y, z]
                            
                        Returns:
                            Float: Distance between the points
                        """
                        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5


                    broken_faces_list = broken_vertices.tolist()

                    # Now check if these broken faces form a fragmented mesh
                    is_frag = is_fragmented(broken_faces_list, tolerance=tolerance)
                    print(f"Are the broken faces fragmented? {is_frag}")

                    print(f"hehe3:{is_frag}")
                    if is_frag:
                        print(" oh shit! sorry the wall has fragments...")
                        print(" marking the scene/room as invalid and moving to next scene/room....")
                        reason_str = "fragments in the wall"
                        invalid_scene_ids.append((scene_unique_id, reason_str))
                        print("-------------------------------------------")
                        print("")
                        skip_scene = True
                        break  # Exit the extras loop; we’ll check skip_scene below
                    
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
                            print(f"All vertices on a plane parallel to YZ. Area: {area_yz}")

                        # Check for single plane cases (XZ-parallel)
                        elif identity== "parallel_to_xz_plane":
                            x_min = np.min(flat_vertices[:, 0])
                            x_max = np.max(flat_vertices[:, 0])
                            z_min = np.min(flat_vertices[:, 2])
                            z_max = np.max(flat_vertices[:, 2])
                            area_xz = (x_max - x_min) * (z_max - z_min)
                            total_projected_area = area_xz
                            print(f"All vertices on a plane parallel  to XZ. Area: {area_xz}")
                        
                        
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
                            print("Vertices span multiple YZ or XZ parallel planes. Total projected area:", total_projected_area)
                    
                        
                        elif (identity=="tilted_across_yz_only" or identity=="tilted_across_xz_only") :           
                            if flat_vertices.shape[0] > 0:
                                # 1. Determine the plane (using the normal of the first broken face)
                                first_broken_face_index = np.where(broken)[0][0]
                                face_normal = mesh.face_normals[first_broken_face_index]
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
                                print(f"Area of the 2D bounding rectangle (projected): {bounding_rectangle_area_2d}")

                            else:
                                print("No broken vertices to calculate a bounding rectangle area.")
                            








                        elif (tilted_yz or tilted_xz) and (yz_parallel or xz_parallel):
                            identity = "tilted_plus_parallel"
                            total_projected_area = 0.0
                            
                            tilted_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Tilted across YZ)" in item[1] or "Vertical Wall (Tilted across XZ)" in item[1])
                            yz_parallel_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                            xz_parallel_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])
                            parallel_count = yz_parallel_count + xz_parallel_count
                            if tilted_count > parallel_count:
                                dominant_orientation = "dominant_tilted"
                                dominant_parallel_type = "n/a" # No dominant parallel type in this case
                                print("nope! add case for this too")
                            elif parallel_count >= tilted_count:
                                dominant_orientation = "dominant_parallel"
                                if yz_parallel_count >= xz_parallel_count:
                                    # Get indices of vertices belonging to YZ parallel faces
                                    yz_indices = [item[0] for item in broken_face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1]]
                                    yz_vertices = flat_vertices[yz_indices]
                                    if yz_vertices.size > 0:  # Ensure there are vertices
                                        y_min_yz = np.min(yz_vertices[:, 1])
                                        y_max_yz = np.max(yz_vertices[:, 1])
                                        z_min_yz = np.min(yz_vertices[:, 2])
                                        z_max_yz = np.max(yz_vertices[:, 2])
                                        area_yz = (y_max_yz - y_min_yz) * (z_max_yz - z_min_yz)
                                        total_projected_area += area_yz
                                        print(f"area of the yz parallel faces is: {total_projected_area}")
                                    else:
                                        print("No vertices found for YZ parallel faces.")
                                elif xz_parallel_count > yz_parallel_count:
                                    # Get indices of vertices belonging to XZ parallel faces
                                    xz_indices = [item[0] for item in broken_face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1]]
                                    xz_vertices = flat_vertices[xz_indices]
                                    if xz_vertices.size > 0:  # Ensure there are vertices
                                        x_min_xz = np.min(xz_vertices[:, 0])
                                        x_max_xz = np.max(xz_vertices[:, 0])
                                        z_min_xz = np.min(xz_vertices[:, 2])
                                        z_max_xz = np.max(xz_vertices[:, 2])
                                        area_xz = (x_max_xz - x_min_xz) * (z_max_xz - z_min_xz)
                                        total_projected_area += area_xz
                                        print(f"area of the xz parallel faces is: {total_projected_area}")
                                    else:
                                        print("No vertices found for XZ parallel faces.")
                            parallel_faces_indices = [item[0] for item in broken_face_orientations if "Parallel" in item[1]]
                            tilted_faces_indices = [item[0] for item in broken_face_orientations if "Tilted" in item[1]]


                            # Calculate projected area of tilted faces (using average normal as before)
                            if tilted_faces_indices:
                                tilted_broken_faces = mesh.faces[tilted_faces_indices]
                                tilted_vertices = mesh.vertices[tilted_broken_faces].reshape(-1, 3)

                                if tilted_vertices.shape[0] > 0:
                                    # 1. Calculate the average normal of the tilted faces
                                    tilted_face_normals = [item[2] for item in broken_face_orientations if item[0] in tilted_faces_indices]
                                    if tilted_face_normals:
                                        average_normal = np.mean(tilted_face_normals, axis=0)
                                        norm = np.linalg.norm(average_normal)
                                        if norm > 0:
                                            average_normal = average_normal / norm
                                        else:
                                            print("Warning: Zero average normal for tilted faces.")
                                            average_normal = np.array([0.0, 1.0, 0.0]) # Default up vector
                                    else:
                                        print("Warning: No normals found for tilted faces.")
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
                                    print(f"Projected area of tilted faces (aligned with average normal): {tilted_projected_area}")
                                    total_projected_area += tilted_projected_area
                                else:
                                    print("No tilted vertices to calculate projected area.")

                            print(f"Total approximate projected area (tilted + parallel): {total_projected_area}")


                        elif tilted_yz and tilted_xz and not yz_parallel and not xz_parallel:
                            identity = "mix_tilted_planes"
                            total_projected_area = 0.0

                            tilted_yz_indices = [item[0] for item in broken_face_orientations if "Tilted across YZ" in item[1]]
                            tilted_xz_indices = [item[0] for item in broken_face_orientations if "Tilted across XZ" in item[1]]

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
                                    print("Warning: Zero average normal for tilted faces.")
                                    average_normal = np.array([0.0, 1.0, 0.0]) # Default up vector

                                # Create a coordinate system aligned with the average normal
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

                            # Calculate projected area for tilted across YZ faces
                            yz_area, num_yz = calculate_projected_area_aligned(tilted_yz_indices, mesh.vertices, mesh.face_normals)
                            print(f"Projected area of tilted (YZ) faces (aligned): {yz_area}")
                            total_projected_area += yz_area

                            # Calculate projected area for tilted across XZ faces
                            xz_area, num_xz = calculate_projected_area_aligned(tilted_xz_indices, mesh.vertices, mesh.face_normals)
                            print(f"Projected area of tilted (XZ) faces (aligned): {xz_area}")
                            total_projected_area += xz_area

                            print(f"Total approximate projected area (tilted YZ + tilted XZ - aligned): {total_projected_area}")




                    #TODO: find good threshold.
                    # i found a wall whhere even whena all faces were combined, it left a bit of thin vertical space.
                    # meaning we should not consider that as a gap for door and window and exclude. we need that area as threshold.
                    # or may be calculate the area of normal door and keep a value lower than that.
                    # it seems doors area is usually above 1.
                    area_tolerance=0.9
                    if total_projected_area>total_broken_area+area_tolerance and not is_frag: # add either conditions like if faces_broken=2 or less.
                        print("there is enough space available for either a wall or a window.")
                        print("--------------------------------------------------")
                        print("now, find where to fit a door." \
                        "for the parallel on any on of the axis, its already working." \
                        "for the one with mixture of parallel faces, find the largest one, and perform as above." \
                        "for the slaned ones... think")
                    #Case1: in any of the plane parallel.
                        if identity=="parallel_to_yz_plane":
                            #operation here...
                            constant_coord = 'x'
                            constant_value = np.mean(flat_vertices[:, 0])  # take mean
                            remaining_coords = flat_vertices[:, 1:3]  # y and z
                            print("parallel_to_yz_plane")
                        
                        elif identity=="parallel_to_xz_plane":
                            constant_coord = 'y'
                            constant_value = np.mean(flat_vertices[:, 1])  # take mean
                            remaining_coords = flat_vertices[:, [0, 2]] 
                            print("parallel_to_xz_plane")

                        #we prioritize the plane with maximum number of vertices, likely only that has the door/window
                        #TODO:#but can be refined.
                        elif identity == "mix_parallel_planes":
                            yz_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                            xz_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])

                            if yz_count >= xz_count:
                                dominant_orientation = "dominant_yz_parallel"
                                constant_coord = 'x'
                                constant_value = np.mean(flat_vertices[:, 0])  # take mean
                                remaining_coords = flat_vertices[:, 1:3]  # y and z
                                print("mixed but prioritiezed parallel_to_yz_plane")

                            elif xz_count > yz_count:
                                constant_coord = 'y'
                                constant_value = np.mean(flat_vertices[:, 1])  # take mean
                                remaining_coords = flat_vertices[:, [0, 2]] 
                                dominant_orientation = "dominant_xz_parallel"
                                print("mixed but prioritiezed parallel_to_xz_plane")
                            # else:
                            #     dominant_orientation = "equal_yz_xz_parallel" # Or handle this case as needed
                            print(f"YZ Parallel Count: {yz_count}, XZ Parallel Count: {xz_count}, Dominant: {dominant_orientation}") # For verification

                        #Case 2: tilted and parallel combination
                        #we prioritize the parallel one and assume it contains the doors/windows.
                        elif identity == "tilted_plus_parallel":
                            tilted_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Tilted across YZ)" in item[1] or "Vertical Wall (Tilted across XZ)" in item[1])
                            yz_parallel_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to YZ Plane)" in item[1])
                            xz_parallel_count = sum(1 for item in broken_face_orientations if "Vertical Wall (Parallel to XZ Plane)" in item[1])
                            parallel_count = yz_parallel_count + xz_parallel_count
                            if tilted_count > parallel_count:
                                dominant_orientation = "dominant_tilted"
                                dominant_parallel_type = "n/a" # No dominant parallel type in this case
                                print("nope! add case for this too")
                            elif parallel_count >= tilted_count:
                                dominant_orientation = "dominant_parallel"
                                if yz_parallel_count > xz_parallel_count:
                                    dominant_parallel_type = "dominant_yz_parallel"
                                    constant_coord = 'x'
                                    constant_value = np.mean(flat_vertices[:, 0])  # take mean
                                    remaining_coords = flat_vertices[:, 1:3]  # y and z
                                    print("parallel and tilted mixed but go with parallel_to_yz_plane")

                                elif xz_parallel_count > yz_parallel_count:
                                    dominant_parallel_type = "dominant_xz_parallel"
                                    constant_coord = 'y'
                                    constant_value = np.mean(flat_vertices[:, 1])  # take mean
                                    remaining_coords = flat_vertices[:, [0, 2]] 
                                    dominant_orientation = "dominant_xz_parallel"
                                    print("parallel and tilted mixed but go with parallel_to_xz_plane")
                            print(f"Tilted Count: {tilted_count}, YZ Parallel Count: {yz_parallel_count}, XZ Parallel Count: {xz_parallel_count}, Dominant Category: {dominant_orientation}, Dominant Parallel Type: {dominant_parallel_type}")
            
                
                                
                        # Step 3: Extract the unique values for the remaining coordinates (y and z or x and z or x and y)
                        unique_remaining_coords = np.unique(remaining_coords, axis=0)

                        # Step 4: Get the min and max for the remaining coordinates
                        min_coord = np.min(unique_remaining_coords, axis=0)
                        max_coord = np.max(unique_remaining_coords, axis=0)

                        # Print the constant coordinate and its value
                        # this gives wall at:
                        print(f"Constant coordinate: {constant_coord} = {constant_value}")


                        # Step 5: Identify the missing gap (door) in the remaining coordinates
                        # We'll check for gaps in the sorted unique coordinates
                        #meaning in that given wall, there is a section void ( so we check the range in which it is void)
                        sorted_remaining_coords = np.sort(unique_remaining_coords, axis=0)
                        print(f"sorted: {sorted_remaining_coords}")


                        #remove duplicacy
                        sorted_remaining_coords = np.unique(sorted_remaining_coords, axis=0)
                        print(f"Unique sorted remaining coords:\n{sorted_remaining_coords}")

                        first_column = sorted_remaining_coords[:, 0]
                        print(f"First column of remaining coords:\n{first_column}")

                        mean_first_column = np.mean(first_column)
                        print(f"Mean of the first column: {mean_first_column}")



                        # Initialize min_1 and max_1 as None
                        min_1 = None
                        max_1 = None

                        distances_from_mean = np.abs(first_column - mean_first_column)

                        # Step 2: Sort the points by how close they are to the mean
                        sorted_indices = np.argsort(distances_from_mean)
                        sorted_points = first_column[sorted_indices]

                        for i in range(len(sorted_points) - 1):
                            point1 = sorted_points[i]
                            point2 = sorted_points[i+1]
                            distance_between_points = np.abs(point1 - point2)
                            
                            if distance_between_points > 0.5:
                                print(f"Found suitable points: {point1} and {point2}")
                                print(f"Distance between them: {distance_between_points}")

                                # Assign min and max
                                if point1 > point2:
                                    max_1 = point1
                                    min_1 = point2
                                else:
                                    max_1 = point2
                                    min_1 = point1

                                print(f"min_1 = {min_1}, max_1 = {max_1}")
                                
                        else:
                            print("No suitable pair of points found with distance > 0.5")



                        # # Loop over the first column values
                        # for val in first_column:
                        #     if val < mean_first_column:
                        #         if (min_1 is None) or (val > min_1):  # Closest value less than mean
                        #             min_1 = val
                        #     elif val > mean_first_column:
                        #         if (max_1 is None) or (val < max_1):  # Closest value greater than mean
                        #             max_1 = val

                        # print(f"Adjusted min_1 (less than mean): {min_1}")
                        # print(f"Adjusted max_1 (greater than mean): {max_1}")


                        # Assuming you're interested in the z-axis (second column)
                        z_coordinates = sorted_remaining_coords[:, 1]
                        sorted_z = np.sort(z_coordinates)
                        differences = np.diff(sorted_z)

                        if differences.size > 0:
                            largest_diff_index = np.argmax(differences)
                            largest_difference = differences[largest_diff_index]

                            # The two points with the largest consecutive difference are at:
                            # sorted_z[largest_diff_index] and sorted_z[largest_diff_index + 1]

                            val1 = sorted_z[largest_diff_index]
                            val2 = sorted_z[largest_diff_index + 1]

                            if val1 < val2:
                                min2 = val1
                                max2 = val2
                            else:
                                min2 = val2
                                max2 = val1

                            print(f"min z value: {min2}")
                            print(f"max z value: {max2}")
                        else:
                            print("Not enough unique points to calculate consecutive differences.")

                        min_2 = min2
                        max_2 = max2


                        # Step 4: Print the results for the gaps in the remaining axes
                        if min_1 is not None and max_1 is not None:
                            print(f" gap found in the first remaining coordinate: from {min_1} to {max_1}")
                        else:
                            print("No  gap found in the first remaining coordinate.")

                        if min_2 is not None and max_2 is not None:
                            print(f" gap found in the second remaining coordinate: from {min_2} to {max_2}")
                        else:
                            print("No  gap found in the second remaining coordinate.")





                                
                        is_door = False

                        if min_2 is not None and min_1 is not None and max_2 is not None and max_1 is not None:
                            if min_2 <= 0.1:  
                                is_door = True
                                print(f"This is a Door, starting from the floor (z_min).")
                            else:
                                is_window=True
                                print(f"This is a Window.")

                            if constant_coord == 'x':
                                        gap_thickness = 0.05  # Example thickness for the gap
                                    
                                        width = max_1 - min_1
                                        height= max_2 - min_2
                                        print(f"width:{width} and height:{height}")
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

                            else:  # constant_coord is 'y' (assuming this is the only other possibility)
                                        gap_thickness = 0.05  # Example thickness for the gap
                                        width = max_1 - min_1
                                        height =max_2 - min_2
                                        print(f"width:{width} and height:{height}")
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
                                        gap_faces = np.array([
                                            [0, 1, 2], [0, 2, 3],   # Front face
                                            [4, 5, 6], [4, 6, 7],   # Back face
                                            [0, 4, 7], [0, 7, 3],   # Bottom face
                                            [1, 5, 6], [1, 6, 2],   # Top face
                                            [0, 4, 7], [0, 7, 3],   # Left face
                                            [3, 2, 6], [3, 6, 7]    # Right face
                                        ])

                            door_window_mesh = trimesh.Trimesh(vertices=gap_vertices, faces=gap_faces)
                            

                            # Creating a simple mesh for door/window from the gap vertices (assuming quad face)
                            print("Shape of gap_vertices:", gap_vertices.shape)
                            print("Content of gap_vertices:\n", gap_vertices)

                            #TODO: validate this: since it will save computation if we don't need to calculate intersection
                            #best_wall_id, iou_score = find_closest_wall_by_vertex_enclosure(obj, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
                            iou_score=0.5#just for fun, discard it.
                            best_wall_id = obj.model_uid
                            print(f"Wall ID: {obj.model_uid}")  # or whatever ID attribute you want
                            

                            if best_wall_id is not None and iou_score > 0.0: # Only process if a wall with sufficient IoU is found
                                wall_id = extra2index.get(best_wall_id, 0)
                                
                                front_face_indices = np.array([0, 1, 2, 3])
                                front_face_vertices = door_window_mesh.vertices[front_face_indices]
                                position = front_face_vertices.mean(axis=0)
                                position_x = position[0]
                                position_y = position[1]
                                position_z = position[2]
                                

                                # Add to the appropriate list (window or door)
                                if is_door:
                                        windows_meshes_to_concatenate.append(door_window_mesh)
                                        print(f"wall id for door (by IoU) is : {wall_id}, IoU: {iou_score}")
                                        door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
                                        print(door_obj.to_language_string())
                                        output_string += door_obj.to_language_string() + '\n'
                                else: #windows 
                                    #windows_meshes_to_concatenate.append(door_window_mesh)
                                    print(f"wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
                                    win_obj = Window(obj_index, wall_id, position_x, position_y, position_z, width, height, "window")
                                    print(win_obj.to_language_string())
                                    output_string += win_obj.to_language_string() + '\n' 
                            else:
                                print("no wall matched to place door or window...")

                        else:
                            print("this wall also doesnt hap gap........................")

                    else:
                        print("this is just a wall and no other elements can be fit into it.")
                        print("--------------------------------------------------")




                    floor_vertices = vertices[np.isclose(vertices[:, 2], z_min, atol=1e-3)]
                    floor_vertices = np.unique(floor_vertices, axis=0)

                    if len(floor_vertices) < 2:
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
                    vertices = mesh.vertices
                    z_min = vertices[:, 2].min()
                    z_max = vertices[:, 2].max()
                    height = z_max - z_min

                    # Use a consistent index for Wall objects, even if they come from the combined list
                    wall = Wall(obj_index, ax, ay, az, bx, by, bz, height, thickness)
                    print(wall.to_language_string())
                    output_string += wall.to_language_string() + '\n'



                elif hasattr(obj, 'model_type') and obj.model_type in ["ExtrusionCustomizedBackgroundWall","CustomizedFeatureWall"]:
                    print("customized wall added to invalid scene ids.....")
                    reason_str="customized wall"
                    invalid_scene_ids.append((scene_unique_id,reason_str))
                    #TODO: validate getting out of loop
                    print("Skipping this scene due to customized wall.")
                    skip_scene = True
                    break  # Exit the extras loop
                #add to invalid one..
                #MAKE THE WHOLE SCENE INAVLID. NO FURTHER PROCESSING FOR OTHER WALL SHOULD HAPPEN.
            else:
                # we are taking walls and dealing with them, anything apart from them in model_type. it is ignored.
                print(f"for {obj.model_type},Skipping language generation. (This is not wall.)")
        else:
            print(f"Warning: Invalid or empty xyz or faces data for object at index {obj_index}. Skipping.")

    if len(wall_meshes_to_concatenate) <2:
        reson_str =f"no sufficient walls, only {len(wall_meshes_to_concatenate)} walls"
        invalid_scene_ids.append((scene_unique_id, "no sufficient walls"))
        print("❌ not sufficient walls.")
        print("moving to next scene....")
        not_suff_wall=True
        



        #         #TODO: whaaaaaaaaaa, might need to adjust
        #         if num_broken>=6: #these are the  walls  that  contain doors and windows

        #             flat_vertices = broken_vertices.reshape(-1, 3)

        #             tolerance = 1e-6  # For comparing floating-point numbers
        #             total_projected_area = 0

        #             # Step 2: Check which coordinate is constant (x, y, or z)
        #             unique_x = np.unique(flat_vertices[:, 0])
        #             unique_y = np.unique(flat_vertices[:, 1])
        #             unique_z = np.unique(flat_vertices[:, 2])



        #             threshold = 0.1  # You can adjust this threshold based on your needs

        #             # Function to check if values are close enough
        #             def is_close(values, threshold):
        #                 return np.ptp(values) < threshold  # ptp = max - min (range)

        #             # Determine which coordinate is (almost) constant
        #             if len(unique_x) == 1 or is_close(unique_x, threshold):
        #                 constant_coord = 'x'
        #                 constant_value = np.mean(flat_vertices[:, 0])  # take mean
        #                 remaining_coords = flat_vertices[:, 1:3]  # y and z
        #             elif len(unique_y) == 1 or is_close(unique_y, threshold):
        #                 constant_coord = 'y'
        #                 constant_value = np.mean(flat_vertices[:, 1])  # take mean
        #                 remaining_coords = flat_vertices[:, [0, 2]]  # x and z
        #             else:
        #                 #TODO: i dont  know what to do , i dont know what to do.
        #                 non_axis_aligned_wall_count += 1
        #                 print("this wall is not axis aligned.......")
        #                 continue
                                    


                            
                    
                            
        #             # Step 3: Extract the unique values for the remaining coordinates (y and z or x and z or x and y)
        #             unique_remaining_coords = np.unique(remaining_coords, axis=0)

        #             # Step 4: Get the min and max for the remaining coordinates
        #             min_coord = np.min(unique_remaining_coords, axis=0)
        #             max_coord = np.max(unique_remaining_coords, axis=0)

        #             # Print the constant coordinate and its value
        #             # this gives wall at:
        #             print(f"Constant coordinate: {constant_coord} = {constant_value}")


        #             # Step 5: Identify the missing gap (door) in the remaining coordinates
        #             # We'll check for gaps in the sorted unique coordinates
        #             #meaning in that given wall, there is a section void ( so we check the range in which it is void)
        #             sorted_remaining_coords = np.sort(unique_remaining_coords, axis=0)
        #             print(f"sorted: {sorted_remaining_coords}")


        #             #remove duplicacy
        #             sorted_remaining_coords = np.unique(sorted_remaining_coords, axis=0)
        #             print(f"Unique sorted remaining coords:\n{sorted_remaining_coords}")

        #             first_column = sorted_remaining_coords[:, 0]
        #             print(f"First column of remaining coords:\n{first_column}")

        #             mean_first_column = np.mean(first_column)
        #             print(f"Mean of the first column: {mean_first_column}")

        #             # Initialize min_1 and max_1 as None
        #             min_1 = None
        #             max_1 = None

        #             # Loop over the first column values
        #             for val in first_column:
        #                 if val < mean_first_column:
        #                     if (min_1 is None) or (val > min_1):  # Closest value less than mean
        #                         min_1 = val
        #                 elif val > mean_first_column:
        #                     if (max_1 is None) or (val < max_1):  # Closest value greater than mean
        #                         max_1 = val

        #             print(f"Adjusted min_1 (less than mean): {min_1}")
        #             print(f"Adjusted max_1 (greater than mean): {max_1}")


        #             # Assuming you're interested in the z-axis (second column)
        #             z_coordinates = sorted_remaining_coords[:, 1]
        #             sorted_z = np.sort(z_coordinates)
        #             differences = np.diff(sorted_z)

        #             if differences.size > 0:
        #                 largest_diff_index = np.argmax(differences)
        #                 largest_difference = differences[largest_diff_index]

        #                 # The two points with the largest consecutive difference are at:
        #                 # sorted_z[largest_diff_index] and sorted_z[largest_diff_index + 1]

        #                 val1 = sorted_z[largest_diff_index]
        #                 val2 = sorted_z[largest_diff_index + 1]

        #                 if val1 < val2:
        #                     min2 = val1
        #                     max2 = val2
        #                 else:
        #                     min2 = val2
        #                     max2 = val1

        #                 print(f"min z value: {min2}")
        #                 print(f"max z value: {max2}")
        #             else:
        #                 print("Not enough unique points to calculate consecutive differences.")

        #             min_2 = min2
        #             max_2 = max2



        #             # Step 4: Print the results for the gaps in the remaining axes
        #             if min_1 is not None and max_1 is not None:
        #                 print(f" gap found in the first remaining coordinate: from {min_1} to {max_1}")
        #             else:
        #                 print("No  gap found in the first remaining coordinate.")

        #             if min_2 is not None and max_2 is not None:
        #                 print(f" gap found in the second remaining coordinate: from {min_2} to {max_2}")
        #             else:
        #                 print("No  gap found in the second remaining coordinate.")

                            
        #             is_door = False

        #             if min_2 is not None and min_1 is not None and max_2 is not None and max_1 is not None:
        #                 if min_2 <= 0.1:  
        #                     is_door = True
        #                     print(f"This is a Door, starting from the floor (z_min).")
        #                 else:
        #                     is_window=True
        #                     print(f"This is a Window.")

        #                 if constant_coord == 'x':
        #                             gap_thickness = 0.05  # Example thickness for the gap
                                
        #                             width = max_1 - min_1
        #                             height= max_2 - min_2
        #                             print(f"width:{width} and height:{height}")
        #                             gap_vertices = np.array([
        #                                 [constant_value, min_1, min_2],
        #                                 [constant_value, max_1, min_2],
        #                                 [constant_value, max_1, max_2],  # Assuming z_max is defined
        #                                 [constant_value, min_1, max_2],  # Assuming z_max is defined
        #                                 [constant_value + gap_thickness, min_1, min_2],
        #                                 [constant_value + gap_thickness, max_1, min_2],
        #                                 [constant_value + gap_thickness, max_1, max_2],
        #                                 [constant_value + gap_thickness, min_1, max_2]
        #                             ])
        #                             gap_faces = np.array([
        #                                 [0, 1, 2], [0, 2, 3],   # Front face
        #                                 [4, 5, 6], [4, 6, 7],   # Back face
        #                                 [0, 4, 7], [0, 7, 3],   # Bottom face
        #                                 [1, 5, 6], [1, 6, 2],   # Top face
        #                                 [0, 1, 5], [0, 5, 4],   # Left face
        #                                 [3, 2, 6], [3, 6, 7]    # Right face
        #                             ])

        #                 else:  # constant_coord is 'y' (assuming this is the only other possibility)
        #                             gap_thickness = 0.05  # Example thickness for the gap
        #                             width = max_1 - min_1
        #                             height =max_2 - min_2
        #                             print(f"width:{width} and height:{height}")
        #                             gap_vertices = np.array([
        #                                 [min_1, constant_value, min_2],
        #                                 [max_1, constant_value, min_2],
        #                                 [max_1, constant_value, max_2],  # Assuming z_max is defined
        #                                 [min_1, constant_value, max_2],  # Assuming z_max is defined
        #                                 [min_1, constant_value + gap_thickness, min_2],
        #                                 [max_1, constant_value + gap_thickness, min_2],
        #                                 [max_1, constant_value + gap_thickness, max_2],
        #                                 [min_1, constant_value + gap_thickness, max_2]
        #                             ])
        #                             gap_faces = np.array([
        #                                 [0, 1, 2], [0, 2, 3],   # Front face
        #                                 [4, 5, 6], [4, 6, 7],   # Back face
        #                                 [0, 4, 7], [0, 7, 3],   # Bottom face
        #                                 [1, 5, 6], [1, 6, 2],   # Top face
        #                                 [0, 4, 7], [0, 7, 3],   # Left face
        #                                 [3, 2, 6], [3, 6, 7]    # Right face
        #                             ])

        #                 door_window_mesh = trimesh.Trimesh(vertices=gap_vertices, faces=gap_faces)
                        

        #                 # Creating a simple mesh for door/window from the gap vertices (assuming quad face)
        #                 print("Shape of gap_vertices:", gap_vertices.shape)
        #                 print("Content of gap_vertices:\n", gap_vertices)

        #                 #TODO: validate this: since it will save computation if we don't need to calculate intersection
        #                 #best_wall_id, iou_score = find_closest_wall_by_vertex_enclosure(obj, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
        #                 iou_score=0.5#just for fun, discard it.
        #                 best_wall_id = obj.model_uid
        #                 print(f"Wall ID: {obj.model_uid}")  # or whatever ID attribute you want
                        

        #                 if best_wall_id is not None and iou_score > 0.0: # Only process if a wall with sufficient IoU is found
        #                     wall_id = extra2index.get(best_wall_id, 0)
                            
        #                     front_face_indices = np.array([0, 1, 2, 3])
        #                     front_face_vertices = door_window_mesh.vertices[front_face_indices]
        #                     position = front_face_vertices.mean(axis=0)
        #                     position_x = position[0]
        #                     position_y = position[1]
        #                     position_z = position[2]
                            

        #                     # Add to the appropriate list (window or door)
        #                     if is_door:
        #                             windows_meshes_to_concatenate.append(door_window_mesh)
        #                             print(f"wall id for door (by IoU) is : {wall_id}, IoU: {iou_score}")
        #                             door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
        #                             print(door_obj.to_language_string())
        #                             output_string += door_obj.to_language_string() + '\n'
        #                     else: #windows (vs mac haha)
        #                         windows_meshes_to_concatenate.append(door_window_mesh)
        #                         print(f"wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
        #                         win_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "window")
        #                         print(win_obj.to_language_string())
        #                         output_string += win_obj.to_language_string() + '\n' 
        #                 else:
        #                     print("no wall matching...")

        #             else:
        #                 print("this wall also doesnt hap gap........................")
        #         else:
        #             print(f"num of broken faces is:{num_broken}")
        #             print("this wall has no doors or windows.")



        #     floor_vertices = vertices[np.isclose(vertices[:, 2], z_min, atol=1e-3)]
        #     floor_vertices = np.unique(floor_vertices, axis=0)

        #     if len(floor_vertices) < 2:
        #         continue  # skip malformed wall

        #     xz = floor_vertices[:, [0, 1]]  # X-Y plane since Z is up
        #     dists = squareform(pdist(xz))
        #     i0, i1 = np.unravel_index(np.argmax(dists), dists.shape)

        #     start = floor_vertices[i0]
        #     end = floor_vertices[i1]
        #     ax, ay, az = start[0], start[1], start[2]
        #     bx, by, bz = end[0], end[1], end[2]

        #     thickness = mesh.bounding_box.primitive.extents.min()
        #     thickness = max(thickness, 0.05)

        #     # Use a consistent index for Wall objects, even if they come from the combined list
        #     wall = Wall(obj_index, ax, ay, az, bx, by, bz, height, thickness)
        #     print(wall.to_language_string())
        #     output_string += wall.to_language_string() + '\n'

        # else:
        #     # we are taking walls and dealing with them, anything apart from them in model_type. it is ignored.
        #     print(f"for {obj.model_type},Skipping language generation. (This is not wall.)")
          
    if skip_scene:
        continue  # Skip rest of the scene including furniture processing

    
    if not_suff_wall:
        continue 
        
    list_of_mesh_to_concatenate = []

    for index, furniture in enumerate(s.bboxes):
        try:
            # 1. Load raw mesh
            mesh = furniture.raw_model()

            # if mesh is None:
            #     print(f"Skipping null mesh at index {index} or load fail problem.")
            #     reason_str = "Failing to load furntiure or its mesh is None"
            #     invalid_scene_ids.append((scene_unique_id, reason_str))
            #     continue

            class_name = furniture.label

            # 2. Apply model-to-scene transform
            vertices = furniture._transform(mesh.vertices)
            mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

            # # Skip empty meshes
            # if mesh.is_empty:
            #     print(f"Skipping empty mesh at index {index} ({class_name})")
            #     continue

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

            print(bbox_obj.to_language_string())
            output_string += bbox_obj.to_language_string() + '\n'

        except Exception as e:
            print(f"[Error] Index {index} - {class_name}: {e}")
            invalid_scene_ids.append((scene_unique_id, e))
            logger.error(f"[Error] Index {index} - {class_name}: {e}")
            continue


    if desired:
        print("------operation before saving the script")
        #only do this if valid:
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
        # print(formatted_output)
        print("formatted in the chronological order of wall, door, window, bbox...")



        #only if valid::
        # -------------Visualization part ------------------
        # only layout represent the scene based on script. other are only for understandiing and validation 
        scene_unique_id= scene_unique_id
        # print(f"scene unique id: {scene_unique_id}")
        room_id = scene_unique_id.split("__")[-1]
        print("room_id:", room_id)
        # Create output directory with room_id
        output_dir = os.path.join("output_meshes", room_id)
        os.makedirs(output_dir, exist_ok=True)

        # to visualize furniture meshes to visualize: yellow color
        if list_of_mesh_to_concatenate:
            concatenated_mesh = trimesh.util.concatenate(list_of_mesh_to_concatenate)
            # Set filename with room_id
            output_filename = os.path.join(output_dir, "concatenated_mesh.obj")
            concatenated_mesh.export(output_filename)
            #print(f"succesffuly concatenated and exported to {output_filename}")
        else:
            print("no valid furnitures meshes to concatenate")
        #to visualize windows and door meshes : sky blue color
        if windows_meshes_to_concatenate:
            concatenated_windows_mesh = trimesh.util.concatenate(windows_meshes_to_concatenate)
            output_filename = os.path.join(output_dir, "concatenated_windows_mesh.obj")
            concatenated_windows_mesh.export(output_filename)
            #print(f"succesffuly concatenated windows and doors and exported to {output_filename}")
        else:
            print("no valid windows meshes to concatenate")
        #to visualize wall meshes : brown color
        if wall_meshes_to_concatenate:
            concatenated_walls_mesh = trimesh.util.concatenate(wall_meshes_to_concatenate)
            output_filename = os.path.join(output_dir, "concatenated_walls_mesh.obj")
            concatenated_walls_mesh.export(output_filename)
            #print(f"succesffuly concatenated walls and exported to {output_filename}")
        else:
            print("no valid walls meshes to concatenate")



        #only if valid:
        import re
        def remove_furniture_bboxes(formatted_output, furnitures_to_remove):
                lines = formatted_output.strip().split('\n')
                removal_lines = []
                remaining_lines = []

                removal_set = set(furnitures_to_remove)

                for line in lines:
                    match = re.match(r'bbox_(\d+)=', line)
                    if match:
                        idx = int(match.group(1))
                        if idx in removal_set:
                            removal_lines.append(line)
                        else:
                            remaining_lines.append(line)
                    else:
                        remaining_lines.append(line)

                removed = '\n'.join(removal_lines)
                cleaned = '\n'.join(remaining_lines)
                return removed, cleaned
                
        #provides furniture which do not collide with walls
        #provide ids of invalid furnitures
        valid_furniture, invalid_ids = filter_furniture_in_scene(wall_meshes_to_concatenate, furniture_meshes, enclosure_threshold=0.7)

        #remove the correspoinding script of the furniture that has collision
        furniture_collided_with_wall_to_remove= invalid_ids
        removed, formatted_output_cleaned = remove_furniture_bboxes(formatted_output, furniture_collided_with_wall_to_remove)
        if not invalid_ids:
            print("✅ All furniture items are considered valid (enclosed by the walls).")
        elif len(invalid_ids)==len(furniture_meshes):
            print(f"all furnitures intersect with wall. so maybe skip")
            reason_str = "all furnitures intersect with wall. "
            invalid_scene_ids.append((scene_unique_id, reason_str))
        else:
            print(f"⚠️ {len(invalid_ids)} out of {len(furniture_meshes)} furniture items are intersect with wall.")
            print(f"✅ Invalid furniture {invalid_ids}  removed")




        #furniture and furniture test

        #remove one of the furniture among the collided one from each pair. 
        #removes the one that is repeated in most as one of the furntiures in the pairs.
        from collections import Counter
        def choose_furnitures_to_remove(collisions):
            # Count how many times each furniture appears in collisions
            counter = Counter()
            for i, j, _, _ in collisions:
                counter[i] += 1
                counter[j] += 1

            # Sort furniture by number of collisions (most collisions first)
            sorted_furnitures = [furniture for furniture, _ in counter.most_common()]

            to_remove = set()
            already_resolved = set()

            for i, j, _, _ in collisions:
                # If this collision already resolved, skip
                if (i in to_remove) or (j in to_remove):
                    continue

                # Remove the one with higher collision count
                if counter[i] >= counter[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

            return list(to_remove)

        #furniture and furniture test
        furniture_furniture_collisions = filter_furniture_furniture_collisions(furniture_meshes)
        if furniture_furniture_collisions:
            print("⚠️ Furniture-Furniture collisions detected.")
            collisions = furniture_furniture_collisions
            furnitures_to_remove = choose_furnitures_to_remove(collisions)
            removed, formatted_output_cleaned = remove_furniture_bboxes(formatted_output_cleaned, furnitures_to_remove)
            print(f"✅ removed {removed} from the script.")
        else:
            print("✅ No undesired furniture-furniture collisions.")

    



        #change the script: bbox1,bbox2,.... ===>bobx and add name.
        #input: formatted_output_cleaned 
        visualize_formatted_output_cleaned=formatted_output_cleaned
        formatted_output_cleaned=re.sub(r'\bbbox_\d+=', 'bbox=', formatted_output_cleaned)

        # # Extract room name from scene_unique_id
        match = re.search(r'__(\w+)-', scene_unique_id)
        room_name = match.group(1) if match else "Unknown"
        formatted_output_cleaned = f"Room({room_name})\n" + formatted_output_cleaned.strip()

    
        #----------------SAVING the script -----------------------------
        output_dir = "SCRIPTS"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name= f"{scene_unique_id}.txt"
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path,"w")as f:
            f.write(formatted_output_cleaned) #TODO:
        print(f"✅ succesfully wrote script to :{output_path}")
        valid_scene_ids.append(scene_unique_id) 
    

        output_dir = "VISUALIZE_SCRIPTS"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name= f"{scene_unique_id}.txt"
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path,"w")as f:
            f.write(visualize_formatted_output_cleaned) #TODO:
        print(f"✅ succesfully wrote script to :{output_path}")
        valid_scene_ids.append(scene_unique_id) 

        log("Comleted Processing scene:", s.scene_id)
        print("")
        print("")
        print("")

    else:
        print("❌ not the desire scene. marked as invalid")
        log("Comleted Processing scene:", s.scene_id)
        print("")
        print("")
        print("")



#----------------SUMMARY-------------------

output_dir = "SUMMARY"
os.makedirs(output_dir, exist_ok=True)

# Invalid scenes with reasons
invalid_output_file_name = "invalid_scene_ids.txt"
invalid_output_path = os.path.join(output_dir, invalid_output_file_name)
with open(invalid_output_path, "w") as f:
    for scene_id, reason in invalid_scene_ids:
        f.write(f"{scene_id} -> {reason}\n")

        

valid_output_file_name= f"valid_scene_ids.txt"
valid_output_path = os.path.join(output_dir, valid_output_file_name)
with open(valid_output_path, "w") as f:
    for valid_id in valid_scene_ids:
        f.write(f"{valid_id}\n")


