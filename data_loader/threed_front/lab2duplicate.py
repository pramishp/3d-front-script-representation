import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from itertools import combinations
import pdb  # Import the debugger

def get_colliding_meshes(meshes, threshold_fraction=0.05, min_count=1):
    from itertools import combinations

    collisions = []

    for i, j in combinations(range(len(meshes)), 2):
        mesh1, mesh2 = meshes[i], meshes[j]

        # AABB check
        bbox1 = mesh1.bounding_box.bounds
        bbox2 = mesh2.bounding_box.bounds
        if (bbox1[1] < bbox2[0]).any() or (bbox2[1] < bbox1[0]).any():
            continue

        try:
            # Check vertices of mesh1 inside mesh2
            mask1_in_2 = mesh2.contains(mesh1.vertices)
            count1_in_2 = mask1_in_2.sum()

            # Check vertices of mesh2 inside mesh1
            mask2_in_1 = mesh1.contains(mesh2.vertices)
            count2_in_1 = mask2_in_1.sum()

            # Total vertices
            total1 = len(mesh1.vertices)
            total2 = len(mesh2.vertices)

            # Intersection condition
            if (count1_in_2 > min_count and count1_in_2 / total1 > threshold_fraction) or \
               (count2_in_1 > min_count and count2_in_1 / total2 > threshold_fraction):

                collisions.append((i, j, count1_in_2, count2_in_1))
                print(f"⚠️ Mesh {i} and Mesh {j} intersect: "
                      f"{count1_in_2} verts from {i} in {j}, "
                      f"{count2_in_1} verts from {j} in {i}")

        except Exception as e:
            print(f"⚠️ Containment test failed between mesh {i} and {j}: {e}")
            continue

    return collisions


output_dir = "more_tests..."
os.makedirs(output_dir, exist_ok=True)


def log(*args):
    print("=" * 10, *args, "=" * 10)

































#not sure how i seem to change the code, and it didn't work so came up with new idea
# just find the center of bounding box of "extras" and calculate its distance to all walls. the nearest one is desired one.
#this works good for a single room as it calculates the nearest one
#when extended to taking door from other room, it was a hassle
def find_closest_wall_by_distance(window_extra, all_extras, walls=["WallInner", "WallOuter", "Front", "Back"]):
    window_mesh = trimesh.Trimesh(vertices=window_extra.xyz, faces=window_extra.faces)
    window_center = window_mesh.bounding_box.centroid
    min_distance = float('inf')
    best_wall_id = None

    for wall_extra in [e for e in all_extras if e.model_type in walls]:
        wall_mesh = trimesh.Trimesh(vertices=wall_extra.xyz, faces=wall_extra.faces)
        wall_center = wall_mesh.bounding_box.centroid
        distance = np.linalg.norm(window_center - wall_center)

        if distance < min_distance:
            min_distance = distance
            best_wall_id = wall_extra.model_uid

    return best_wall_id, min_distance if best_wall_id is not None else float('inf')

#








import trimesh
import numpy as np

def calculate_mesh_overlap_score(mesh1, mesh2):
    """
    Calculates a score representing the overlap between two meshes.
    This score considers the volume of the intersection over the volumes
    of the individual meshes.

    Args:
        mesh1 (trimesh.Trimesh): The first mesh.
        mesh2 (trimesh.Trimesh): The second mesh.

    Returns:
        float: A score between 0.0 and 1.0 representing the overlap.
               0.0 means no overlap, 1.0 means perfect overlap (identical meshes).
    """
    if mesh1.is_empty or mesh2.is_empty:
        return 0.0

    intersection = trimesh.intersections.mesh_mesh_collision(mesh1, mesh2)

    if intersection is None or intersection.volume <= 1e-9:  # Avoid division by zero for tiny intersections
        return 0.0

    volume_intersection = intersection.volume
    volume_mesh1 = mesh1.volume
    volume_mesh2 = mesh2.volume

    # Calculate a combined overlap score
    overlap_score = volume_intersection / (volume_mesh1 + volume_mesh2 - volume_intersection + 1e-9)
    return np.clip(overlap_score, 0.0, 1.0)

def find_closest_mesh_by_overlap(target_mesh_extra, all_mesh_extras, candidate_types=None):
    """
    Finds the mesh in a list that has the highest volume overlap with a target mesh.

    Args:
        target_mesh_extra (object): An object containing the target mesh data (must have a .xyz and .faces attribute).
        all_mesh_extras (list of objects): A list of objects, each containing mesh data
                                          (must have a .xyz and .faces attribute).
        candidate_types (list of str, optional): A list of model_type strings to consider.
                                                 If None, all meshes in all_mesh_extras are considered.
                                                 Defaults to None.

    Returns:
        tuple: A tuple containing the model_uid of the closest mesh and the maximum overlap score.
               Returns (None, 0.0) if no suitable candidate meshes are found.
    """
    target_mesh = trimesh.Trimesh(vertices=target_mesh_extra.xyz, faces=target_mesh_extra.faces)

    max_overlap = 0.0
    best_match_id = None

    candidate_meshes = all_mesh_extras
    if candidate_types:
        candidate_meshes = [e for e in all_mesh_extras if getattr(e, 'model_type', None) in candidate_types]

    if not candidate_meshes:
        return None, 0.0

    for candidate_extra in candidate_meshes:
        candidate_mesh = trimesh.Trimesh(vertices=candidate_extra.xyz, faces=candidate_extra.faces)
        overlap_score = calculate_mesh_overlap_score(target_mesh, candidate_mesh)

        if overlap_score > max_overlap:
            max_overlap = overlap_score
            best_match_id = getattr(candidate_extra, 'model_uid', None)

    return best_match_id, max_overlap





















#works like charm.. for placing door from one room to others.

def is_point_in_bbox(point, bbox_min, bbox_max, tolerance=1e-6):
    """Checks if a point is within a bounding box with a small tolerance."""
    return all(bbox_min[i] - tolerance <= point[i] <= bbox_max[i] + tolerance for i in range(3))

def calculate_vertex_enclosure_score(mesh1, mesh2):
    """Calculates the percentage of vertices of mesh1 inside the bounding box of mesh2."""
    bbox_min2, bbox_max2 = mesh2.bounding_box.bounds
    enclosed_count = 0
    for vertex in mesh1.vertices:
        if is_point_in_bbox(vertex, bbox_min2, bbox_max2):
            enclosed_count += 1
    return enclosed_count / len(mesh1.vertices) if len(mesh1.vertices) > 0 else 0

def find_closest_wall_by_vertex_enclosure(window_extra, all_extras, walls=["Front", "WallInner", "WallOuter", "Back"]):
    window_mesh = trimesh.Trimesh(vertices=window_extra.xyz, faces=window_extra.faces)

    max_overlap_score = 0.0
    best_wall_id = None

    candidate_walls = [e for e in all_extras if e.model_type in walls]

    if not candidate_walls:
        return None, 0.0

    for wall_extra in candidate_walls:
        wall_mesh = trimesh.Trimesh(vertices=wall_extra.xyz, faces=wall_extra.faces)

        # Calculate the percentage of window vertices inside the wall's bbox
        score1 = calculate_vertex_enclosure_score(window_mesh, wall_mesh)

        # Calculate the percentage of wall vertices inside the window's bbox
        score2 = calculate_vertex_enclosure_score(wall_mesh, window_mesh)

        # Combine the scores (e.g., average)
        overlap_score = (score1 + score2) / 2.0

        if overlap_score > max_overlap_score:
            max_overlap_score = overlap_score
            best_wall_id = wall_extra.model_uid

    return best_wall_id, max_overlap_score












def find_closest_wall_by_iou(window_extra, all_extras, walls=["WallInner", "Front", "Back"]):
    window_mesh = trimesh.Trimesh(vertices=window_extra.xyz, faces=window_extra.faces)
    window_aabb = window_mesh.bounding_box.bounds
    window_min = window_aabb[0]
    window_max = window_aabb[1]

    window_min_xy = window_min[[0, 1]]
    window_max_xy = window_max[[0, 1]]
    window_min_z = window_min[2]
    window_max_z = window_max[2]

    max_score = 0
    best_wall_id = None

    for _, wall_extra in enumerate([e for e in all_extras if e.model_type in walls]):
        wall_mesh = trimesh.Trimesh(vertices=wall_extra.xyz, faces=wall_extra.faces)
        wall_aabb = wall_mesh.bounding_box.bounds
        wall_min = wall_aabb[0]
        wall_max = wall_aabb[1]

        wall_min_xy = wall_min[[0, 1]]
        wall_max_xy = wall_max[[0, 1]]
        wall_min_z = wall_min[2]
        wall_max_z = wall_max[2]

        # 2D IoU (XY)
        intersect_min_xy = np.maximum(window_min_xy, wall_min_xy)
        intersect_max_xy = np.minimum(window_max_xy, wall_max_xy)
        intersect_dims_xy = np.maximum(intersect_max_xy - intersect_min_xy, 0)
        intersection_area_xy = np.prod(intersect_dims_xy)

        window_area = np.prod(window_max_xy - window_min_xy)
        wall_area = np.prod(wall_max_xy - wall_min_xy)
        union_area = window_area + wall_area - intersection_area_xy
        iou_xy = intersection_area_xy / union_area if union_area > 0 else 0

        # Z-overlap ratio
        z_overlap = max(0, min(window_max_z, wall_max_z) - max(window_min_z, wall_min_z))
        z_total = max(window_max_z, wall_max_z) - min(window_min_z, wall_min_z)
        z_overlap_ratio = z_overlap / z_total if z_total > 0 else 0

        # Weighted score (tweak weights as needed)
        score = iou_xy * 0.7 + z_overlap_ratio * 0.3

        if score > max_score:
            max_score = score
            best_wall_id = wall_extra.model_uid

    return best_wall_id, max_score




# Paths
#TODO: uncomment
#path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
path_to_3d_front_dataset_directory = "/home/ajad/Desktop/codes/LISA"
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
transform2 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])



all_doors_across_scenes = []
door_scene_map = {}

# First, let's collect all door objects from all scenes
for scene in d.scenes:
    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
            all_doors_across_scenes.append(extra)
            door_scene_map[extra.model_uid] = scene.scene_id

print("\nDoor IDs with Corresponding Scene IDs:")
for door_id, scene_id in door_scene_map.items():
    print(f"Door ID: {door_id}, Scene ID: {scene_id}")


for s in d.scenes:  #Library-4425
    if s.scene_id != "Library-4425": #LivingDiningRoom-3474
        continue  #MasterBedroom-5863

    log("Processing scene:", s.scene_id)
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}

    output_string =""
   
    windows_meshes_to_concatenate =[] # to visualize windows_meshes with blue color, but here the walls are also added to it.
    wall_meshes=[] # contains all meshes of walls, to check intersection with all furnitures
    furniture_meshes =[] #contains all furnitures meshes to compare intersection with walls

    wall_meshes_with_index = []


    all_doors_across_scenes = []
    # First, let's collect all door objects from all scenes EXCEPT the current one
    for other_scene in d.scenes:
        if other_scene.scene_id != s.scene_id:
            for extra in other_scene.extras:
                if hasattr(extra, 'model_type') and extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
                    all_doors_across_scenes.append(extra)

    

    # Now, iterate through the extras in the current scene AND the collected doors from other scenes
    all_relevant_objects = list(s.extras) + all_doors_across_scenes


    for obj_index, obj in enumerate(all_relevant_objects):
        mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
        mesh.apply_transform(transform)

        vertices = mesh.vertices
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        height = z_max - z_min

        if hasattr(obj, 'model_type') and obj.model_type in ["WallInner", "WallOuter", "Front", "Back"]:
            #windows_meshes_to_concatenate.append(mesh)
            wall_meshes_with_index.append((mesh, obj_index)) # Store mesh and its original index
            wall_meshes.append(mesh)





            # Get the broken face indices
            broken = trimesh.repair.broken_faces(mesh)
            num_broken = len(broken)
            print(f"Found {num_broken} broken faces for wall with index {obj_index}.")

                        # Extract the broken face indices
            broken_faces = mesh.faces[broken]

            if num_broken>=6:

                # Get the corresponding vertex positions for the broken faces
                broken_vertices = mesh.vertices[broken_faces]

                # Print the broken faces indices and vertices
                print(f"Broken face indices: {broken}")
                print(f"Broken vertices coordinates: \n{broken_vertices}")

                import numpy as np
                import trimesh

                # Assuming broken_faces and broken_vertices are already obtained
                # broken_faces = mesh.faces[broken]
                # broken_vertices = mesh.vertices[broken_faces]

                # Step 1: Flatten the broken_vertices to a 2D array (x, y, z) for each face
                # broken_vertices is of shape (num_faces, 3, 3), so flatten it for easy processing
                flat_vertices = broken_vertices.reshape(-1, 3)

                # Step 2: Check which coordinate is constant (x, y, or z)
                unique_x = np.unique(flat_vertices[:, 0])
                unique_y = np.unique(flat_vertices[:, 1])
                unique_z = np.unique(flat_vertices[:, 2])

                # Determine which coordinate is constant by checking the unique values
                if len(unique_x) == 1:
                    # x is constant
                    constant_coord = 'x'
                    constant_value = unique_x[0]
                    remaining_coords = flat_vertices[:, 1:3]  # Only y and z values
                elif len(unique_y) == 1:
                    # y is constant
                    constant_coord = 'y'
                    constant_value = unique_y[0]
                    remaining_coords = flat_vertices[:, [0, 2]]  # Only x and z values
                elif len(unique_z) == 1:
                    # z is constant
                    constant_coord = 'z'
                    constant_value = unique_z[0]
                    remaining_coords = flat_vertices[:, [0, 1]]  # Only x and y values
                else:
                    raise ValueError("No constant coordinate found! This should not happen.")

                        
                # Step 3: Extract the unique values for the remaining coordinates (y and z or x and z or x and y)
                unique_remaining_coords = np.unique(remaining_coords, axis=0)

                # Step 4: Get the min and max for the remaining coordinates
                min_coord = np.min(unique_remaining_coords, axis=0)
                max_coord = np.max(unique_remaining_coords, axis=0)

                # Print the constant coordinate and its value
                # this gives wall at:
                print(f"Constant coordinate: {constant_coord} = {constant_value}")

                # Print the boundaries for the remaining coordinates (y, z, or x, y)
                #this gives the coodinates that form the wall
                if constant_coord == 'x':
                    print(f"Minimum y: {min_coord[0]}")
                    print(f"Maximum y: {max_coord[0]}")
                    print(f"Minimum z: {min_coord[1]}")
                    print(f"Maximum z: {max_coord[1]}")
                elif constant_coord == 'y':
                    print(f"Minimum x: {min_coord[0]}")
                    print(f"Maximum x: {max_coord[0]}")
                    print(f"Minimum z: {min_coord[1]}")
                    print(f"Maximum z: {max_coord[1]}")
                else:  # constant_coord == 'z'
                    print(f"Minimum x: {min_coord[0]}")
                    print(f"Maximum x: {max_coord[0]}")
                    print(f"Minimum y: {min_coord[1]}")
                    print(f"Maximum y: {max_coord[1]}")



                # Step 5: Identify the missing gap (door) in the remaining coordinates
                # We'll check for gaps in the sorted unique coordinates
                #meaning in that given wall, there is a section void ( so we check the range in which it is void)
                sorted_remaining_coords = np.sort(unique_remaining_coords, axis=0)



                # Identify missing gap in the remaining two axes (y, z or x, y, or x, z)
                door_min_1 = None
                door_max_1 = None
                door_min_2 = None
                door_max_2 = None
                threshold = 1  # Threshold for gap detection (adjust as needed)

                # Step 1: Separate the coordinates based on which one is constant
                if constant_coord == 'x':
                    remaining_coords_1 = sorted_remaining_coords[:, 0]  # y values
                    remaining_coords_2 = sorted_remaining_coords[:, 1]  # z values
                elif constant_coord == 'y':
                    remaining_coords_1 = sorted_remaining_coords[:, 0]  # x values
                    remaining_coords_2 = sorted_remaining_coords[:, 1]  # z values
                else:  # constant_coord == 'z'
                    remaining_coords_1 = sorted_remaining_coords[:, 0]  # x values
                    remaining_coords_2 = sorted_remaining_coords[:, 1]  # y values

                # Step 2: Identify the gap for the first remaining coordinate
                for i in range(1, len(remaining_coords_1)):
                    if remaining_coords_1[i] > remaining_coords_1[i-1] + threshold:
                        door_min_1 = remaining_coords_1[i-1]
                        door_max_1 = remaining_coords_1[i]
                        break

                # Step 3: Identify the gap for the second remaining coordinate
                for i in range(1, len(remaining_coords_2)):
                    if remaining_coords_2[i] > remaining_coords_2[i-1] + threshold:
                        door_min_2 = remaining_coords_2[i-1]
                        door_max_2 = remaining_coords_2[i]
                        break

                # Step 4: Print the results for the gaps in the remaining axes
                if door_min_1 is not None and door_max_1 is not None:
                    print(f"Door gap found in the first remaining coordinate: from {door_min_1} to {door_max_1}")
                else:
                    print("No door gap found in the first remaining coordinate.")

                if door_min_2 is not None and door_max_2 is not None:
                    print(f"Door gap found in the second remaining coordinate: from {door_min_2} to {door_max_2}")
                else:
                    print("No door gap found in the second remaining coordinate.")

                            # **Step to check if it starts from the floor (z_min)**
                is_door = False
                if door_min_2 is not None and door_min_2 <= z_min + 0.1:  # Check if gap is near the floor
                    is_door = True
                    print(f"This is a Door, starting from the floor (z_min).")
                else:
                    print(f"This is a Window.")

                
                # **Create Mesh from Gap Coordinates:**
                # Make a mesh for door/window (with a basic rectangular mesh example)
                gap_vertices = np.array([[door_min_1, door_min_2, z_min], [door_max_1, door_min_2, z_min],
                                        [door_max_1, door_max_2, z_min], [door_min_1, door_max_2, z_min]])

                # Creating a simple mesh for door/window from the gap vertices (assuming quad face)
                door_window_faces = np.array([[0, 1, 2], [0, 2, 3]])  # Two triangles for the quad
                door_window_mesh = trimesh.Trimesh(vertices=gap_vertices, faces=door_window_faces)


                windows_meshes_to_concatenate.append(door_window_mesh)
                door_window_mesh.apply_transform(transform)

                bbox = door_window_mesh.bounding_box
                min_corner, max_corner = door_window_mesh.bounds
                win_pos_x, win_pos_y, win_pos__z = ((min_corner + max_corner) / 2).tolist()
                win_width = max(max_corner[1] - min_corner[1], max_corner[0] - min_corner[0])
                win_height = max_corner[2] - min_corner[2]
                center = bbox.centroid
                dims = bbox.extents
                position_x, position_y, position_z = center.tolist()
                width = np.max(dims[:2])
                height = dims[2]

                # Add to the appropriate list (window or door)
                if is_door:
                        windows_meshes_to_concatenate.append(door_window_mesh)
                        print(f"wall id for door (by IoU) is : {wall_id}, IoU: {iou_score}")
                        door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
                        print(door_obj.to_language_string())
                        output_string += door_obj.to_language_string() + '\n'
                else: #windows (vs mac haha)
                        windows_meshes_to_concatenate.append(door_window_mesh)
                        print(f"wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
                        win_obj = Window(obj_index, wall_id, win_pos_x, win_pos_y, win_pos__z, win_width, win_height, "window")
                        print(win_obj.to_language_string())
                        output_string += win_obj.to_language_string() + '\n' 


            else:
                print("this wall has no doors or windows.")









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
            thickness = max(thickness, 0.05)

            # Use a consistent index for Wall objects, even if they come from the combined list
            wall = Wall(obj_index, ax, ay, az, bx, by, bz, height, thickness)
            print(wall.to_language_string())
            output_string += wall.to_language_string() + '\n'

        elif hasattr(obj, 'model_type') and obj.model_type in ["Window", "Door", "interiorDoor", "entryDoor"]:
            # bbox = mesh.bounding_box
            # min_corner, max_corner = mesh.bounds
            # win_pos_x, win_pos_y, win_pos__z = ((min_corner + max_corner) / 2).tolist()
            # win_width = max(max_corner[1] - min_corner[1], max_corner[0] - min_corner[0])
            # win_height = max_corner[2] - min_corner[2]
            # center = bbox.centroid
            # dims = bbox.extents
            # rot_matrix = bbox.primitive.transform[:3, :3]
            # position_x, position_y, position_z = center.tolist()
            # width = np.max(dims[:2])
            # height = dims[2]

 
            best_wall_id, iou_score = find_closest_wall_by_vertex_enclosure(obj, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])

            if best_wall_id is not None and iou_score > 0.0: # Only process if a wall with sufficient IoU is found
                wall_id = extra2index.get(best_wall_id, 0)
                mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
                mesh.apply_transform(transform)

                bbox = mesh.bounding_box
                min_corner, max_corner = mesh.bounds
                win_pos_x, win_pos_y, win_pos__z = ((min_corner + max_corner) / 2).tolist()
                win_width = max(max_corner[1] - min_corner[1], max_corner[0] - min_corner[0])
                win_height = max_corner[2] - min_corner[2]
                center = bbox.centroid
                dims = bbox.extents
                position_x, position_y, position_z = center.tolist()
                width = np.max(dims[:2])
                height = dims[2]

                if obj.model_type == "Window":
                    windows_meshes_to_concatenate.append(mesh)
                    print(f"wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
                    win_obj = Window(obj_index, wall_id, win_pos_x, win_pos_y, win_pos__z, win_width, win_height, "window")
                    print(win_obj.to_language_string())
                    output_string += win_obj.to_language_string() + '\n'
                else:
                    windows_meshes_to_concatenate.append(mesh)
                    print(f"wall id for door (by IoU) is : {wall_id}, IoU: {iou_score}")
                    door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
                    print(door_obj.to_language_string())
                    output_string += door_obj.to_language_string() + '\n'
            else:
                print(f"No suitable wall found by IoU for {obj.model_type} with UID {obj.model_uid}. Skipping language generation.")

    # ... (rest of your code for furniture processing and output)

            # best_wall_id, _ = find_closest_wall_within_threshold(obj, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
            # if best_wall_id is not None:
            #     wall_id = extra2index.get(best_wall_id, 0)

            #     if obj.model_type == "Window":
            #         windows_meshes_to_concatenate.append(mesh)
            #         print(f"wall id for windows is:{wall_id}")
            #         win_obj = Window(obj_index, wall_id, win_pos_x, win_pos_y, win_pos__z, win_width, win_height, "window")
            #         print(win_obj.to_language_string())
            #         output_string += win_obj.to_language_string() + '\n'
            #     else:
            #         windows_meshes_to_concatenate.append(mesh)
            #         print(f"wall id for door is : {wall_id}")
            #         door_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "door")
            #         print(door_obj.to_language_string())
            #         output_string += door_obj.to_language_string() + '\n'
            # else:
            #     print(f"No close wall found within the threshold for {obj.model_type} with UID {obj.model_uid}. Skipping language generation.")




        wall_x_coordinate = -4.79901
        door_y_min = -4.67901
        door_y_max = -3.65023
        door_z_min = 0.0
        door_z_max = 2.1
        door_thickness = 0.1

        door_vertices = np.array([
            [wall_x_coordinate, door_y_min, door_z_min],
            [wall_x_coordinate, door_y_max, door_z_min],
            [wall_x_coordinate, door_y_max, door_z_max],
            [wall_x_coordinate, door_y_min, door_z_max],
            [wall_x_coordinate + door_thickness, door_y_min, door_z_min],
            [wall_x_coordinate + door_thickness, door_y_max, door_z_min],
            [wall_x_coordinate + door_thickness, door_y_max, door_z_max],
            [wall_x_coordinate + door_thickness, door_y_min, door_z_max]
        ])

        door_faces = np.array([
            [0, 1, 2], [0, 2, 3],   # Front face
            [4, 5, 6], [4, 6, 7],   # Back face
            [0, 4, 7], [0, 7, 3],   # Bottom face
            [1, 5, 6], [1, 6, 2],   # Top face
            [0, 1, 5], [0, 5, 4],   # Left face
            [3, 2, 6], [3, 6, 7]    # Right face
        ])

        door_mesh = trimesh.Trimesh(vertices=door_vertices, faces=door_faces)
        windows_meshes_to_concatenate.append(door_mesh)











    # for i, extra in enumerate(s.extras):
    #     mesh = trimesh.Trimesh(vertices=extra.xyz, faces=extra.faces)
    #     mesh.apply_transform(transform)

    #     vertices = mesh.vertices
    #     z_min = vertices[:, 2].min()
    #     z_max = vertices[:, 2].max()
    #     height = z_max - z_min

    #     if extra.model_type in ["WallInner","WallOuter","Front","Back"]: # "WallOuter","WallBottom","WallTop"
    #         windows_meshes_to_concatenate.append(mesh)
    #         # Find two most distant floor-level vertices in XY
    #         wall_meshes.append(mesh)
    #         floor_vertices = vertices[np.isclose(vertices[:, 2], z_min, atol=1e-3)]
    #         floor_vertices = np.unique(floor_vertices, axis=0)

    #         if len(floor_vertices) < 2:
    #             continue  # skip malformed wall

    #         xz = floor_vertices[:, [0, 1]]  # X-Y plane since Z is up
    #         dists = squareform(pdist(xz))
    #         i0, i1 = np.unravel_index(np.argmax(dists), dists.shape)

    #         start = floor_vertices[i0]
    #         end = floor_vertices[i1]
    #         ax, ay, az = start[0], start[1], start[2]
    #         bx, by, bz = end[0], end[1], end[2]

    #         thickness = mesh.bounding_box.primitive.extents.min()
    #         thickness = max(thickness, 0.05)

    #         wall = Wall(i, ax, ay, az, bx, by, bz, height, thickness)
    #         # print(f"wall_id:{i}-model_uid:{extra.model_uid}")
    #         print(wall.to_language_string())
    #         output_string +=wall.to_language_string()+'\n'

        
    #     elif extra.model_type in ["Window", "Door","interiorDoor","entryDoor"]: #interiorDoor, entryDoor, outdoor,
    #         bbox=mesh.bounding_box
    #         min_corner, max_corner = mesh.bounds
    #         win_pos_x, win_pos_y, win_pos__z =((min_corner+max_corner)/2).tolist()
    #         win_width =max(max_corner[1]-min_corner[1], max_corner[0]-min_corner[0])
    #         win_height =max_corner[2]-min_corner[2]
    #         center = bbox.centroid
    #         dims=bbox.extents 
    #         rot_matrix =bbox.primitive.transform[:3,:3]
    #         position_x, position_y, position_z = center.tolist()
    #         width = np.max(dims[:2]) #todo
    #         height = dims[2]  # since Z is up
    #         find_closest_wall_within_threshold
    #         best_wall_id, _ = find_closest_wall_by_distance(extra, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
    #         #best_wall_id, _ = find_closest_wall_by_distance(extra, s.extras, walls=["Front", "WallInner", "WallOuter", "Back"])
    #         #best_wall_id, _ = find_closest_wall_by_iou(extra, s.extras, walls=["Front", "WallInner" "WallOuter" "Back"])
    #         wall_id = extra2index.get(best_wall_id, 0)

    #         if extra.model_type == "Window":
    #             windows_meshes_to_concatenate.append(mesh)
    #             print(f"wall id for windows is:{wall_id}")
    #             obj = Window(i, wall_id, win_pos_x, win_pos_y, win_pos__z, win_width, win_height, "window")
    #         else:
    #             windows_meshes_to_concatenate.append(mesh)
    #             print(f"wall id for door is : {wall_id}")
    #             obj = Door(i, wall_id, position_x, position_y, position_z, width, height, "door")
    #         print(obj.to_language_string())
    #         output_string +=obj.to_language_string()+'\n'









    list_of_mesh_to_concatenate=[]

    for index, furniture in enumerate(s.bboxes):
        # 1. Load raw mesh
        mesh = furniture.raw_model()
        if mesh is not None:

            class_name = furniture.label
            # 2. Apply model-to-scene transform
            vertices = furniture._transform(mesh.vertices)
            mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

            # 3. Apply Z-up transformation (if needed) (needed i guess, else the furnitures are outside the layout formed from extras!)
            mesh.apply_transform(transform)
            furniture_meshes.append(mesh)

            list_of_mesh_to_concatenate.append(mesh)

            bbox = mesh.bounding_box
            #center = mesh.centroid
            center = bbox.primitive.transform[:3,3]
            # center = bbox.centroid
            dims = bbox.extents  # dx, dy, dz

            # Extract heading (rotation around Z axis)
            # Compute rotation matrix from bounding box
            rot_matrix = bbox.primitive.transform[:3, :3]
            # Heading = angle between bbox local X-axis and world X-Y

            heading = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # yaw in radians

            # id: int
            # class_name: str
            # position_x: float
            # position_y: float
            # position_z: float
            # angle_z: float
            # scale_x: float
    
    
    
            # scale_y: float
            # scale_z: float
            # entity_label: str = "bbox"
            bbox = Bbox(index, class_name, center[0], center[1], center[2], heading, dims[0], dims[1], dims[2])
            print(bbox.to_language_string())
            output_string +=bbox.to_language_string()+'\n'

    if list_of_mesh_to_concatenate:
        concatenated_mesh = trimesh.util.concatenate(list_of_mesh_to_concatenate)
        output_filename = "concatenated_mesh.obj"
        concatenated_mesh.export(output_filename)
        #print(f"succesffuly concatenated and exported to {output_filename}")
    else:
        print("no valid furnitures meshes to concatenate")

    if windows_meshes_to_concatenate:
        concatenated_windows_mesh = trimesh.util.concatenate(windows_meshes_to_concatenate)
        output_filename = "concatenated_windows_mesh.obj"
        concatenated_windows_mesh.export(output_filename)
        #print(f"succesffuly concatenated windows and exported to {output_filename}")
    else:
        print("no valid windows meshes to concatenate")




    
    #TODO: change this
    scene_id_hardcoded= "00004f89-9aa5-43c2-ae3c-129586be8aaa"
    room_id = s.scene_id
    output_file_name= f"{scene_id_hardcoded}__{room_id}.txt"
    output_path = os.path.join(output_dir, output_file_name)

    with open(output_path,"w")as f:
        f.write(output_string)
    
    print(f"succesfully wrote combined output to :{output_path}")



    furniture_wall_collisions= get_colliding_meshes(furniture_meshes+wall_meshes)
    scene_has_invalid_furniture = False
    for i, j, c1, c2 in furniture_wall_collisions:
        # i < len(furniture_meshes) means it's a furniture-furniture pair, skip
        if (i < len(furniture_meshes) and j >= len(furniture_meshes)) or (j < len(furniture_meshes) and i >= len(furniture_meshes)):
            print(f"❌ Furniture {i} collides with wall {j}")
            scene_has_invalid_furniture = True
            break

    if scene_has_invalid_furniture:
        print("❌ Skipping scene due to invalid furniture-wall collision.")
        continue
    else:
        print("✅ No undesired furniture-wall collisions.")


    if list_of_mesh_to_concatenate:
        collisions = get_colliding_meshes(list_of_mesh_to_concatenate)
        if collisions:
            print(f"❌ Found {len(collisions)} furniture-furniture collisions in scene.")
            continue  # Skip saving
        else:
            print("✅ No furniture-furniture mesh collisions.")

    

