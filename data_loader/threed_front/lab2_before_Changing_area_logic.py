import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from itertools import combinations
import pdb  # Import the debugger
import gc

#TODO:
# 1. for the scenes whose wall is not aligned to the axis.
#2. find better estimate for furniture and wall intersection.
#since, objects can be large and cross the wall, and not many vertices may fall on the wall. it is a problem.




#-----------FURNITURE AND WALL COLLISION DETECTION-------------
def is_point_in_bbox(point, bbox_min, bbox_max, tolerance=1e-6):
    """Checks if a point is within a bounding box with a small tolerance."""
    return all(bbox_min[i] - tolerance <= point[i] <= bbox_max[i] + tolerance for i in range(3))

def calculate_vertex_enclosure_score(mesh1, mesh2):
    """Calculates the percentage of vertices of mesh1 inside the bounding box of mesh2."""
    bbox_min2, bbox_max2 = mesh2.bounding_box.bounds
    enclosed_count = sum(is_point_in_bbox(vertex, bbox_min2, bbox_max2) for vertex in mesh1.vertices)
    return enclosed_count / len(mesh1.vertices) if mesh1.vertices.size > 0 else 0










#TODO: figure out the threshold
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
        return []  # No walls, exclude all furniture

    # Create the combined wall mesh
    combined_wall_mesh = trimesh.Trimesh(
        vertices=np.vstack(combined_vertices),
        faces=np.vstack(combined_faces),
        process=False
    )

    # Now check each furniture against the combined wall mesh
    valid_furniture = []
    for i, furniture_mesh in enumerate(furniture_meshes):
        try:
            # Create the furniture mesh object
            mesh = trimesh.Trimesh(vertices=furniture_mesh.vertices, faces=furniture_mesh.faces, process=False)

            # Calculate how well the furniture mesh is enclosed by the combined wall mesh
            score = calculate_vertex_enclosure_score(mesh, combined_wall_mesh)

            # If the score is above the threshold, consider it valid
            if score >= enclosure_threshold:
                valid_furniture.append(furniture_mesh)
            else:
                # Print the collision details if the furniture doesn't meet the threshold
                print(f"⚠️ Furniture {i} collides with walls, enclosure score: {score:.2f}")

        except Exception as e:
            print(f"⚠️ Error processing furniture mesh {i}: {e}")
            continue

    # Clean up memory after the check
    del combined_wall_mesh
    gc.collect()  # Trigger garbage collection to free memory

    return valid_furniture










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
                    print(f"⚠️ Furniture {i} overlaps with Furniture {j}, score_i_in_j={score_ij:.2f}, score_j_in_i={score_ji:.2f}")
                    collisions.append((i, j, score_ij, score_ji))

            except Exception as e:
                print(f"⚠️ Error comparing Furniture {i} and Furniture {j}: {e}")
                continue

    return collisions









def log(*args):
    print("=" * 10, *args, "=" * 10)





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


invalid_scene_ids = []
valid_scene_ids=[]



for s in d.scenes:  #Library-4425
    json_filename = s.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{s.scene_id}"
    print(f"scene  id is: {scene_unique_id}")

    if s.scene_id != "Hallway-1213": #LivingDiningRoom-3474
        continue  #MasterBedroom-5863

    log("Processing scene:", s.scene_id)
    print("")
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}

    output_string =""
   
    windows_meshes_to_concatenate =[] # to visualize windows_meshes with blue color, but here the walls are also added to it.
    wall_meshes_to_concatenate=[] # contains all meshes of walls, to check intersection with all furnitures
    furniture_meshes =[] #contains all furnitures meshes to compare intersection with walls

    wall_meshes_with_index = []


    all_relevant_objects = list(s.extras)

    non_axis_aligned_wall_count = 0
    for obj_index, obj in enumerate(all_relevant_objects):
        mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
        mesh.apply_transform(transform)

        vertices = mesh.vertices
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        height = z_max - z_min

        
        if hasattr(obj, 'model_type') and obj.model_type in ["CustomizedFeatureWall","WallInner", "WallOuter", "Front", "Back","ExtrusionCustomizedBackgroundWall",]:
            wall_meshes_to_concatenate.append(mesh)
            wall_meshes_with_index.append((mesh, obj_index)) # Store mesh and its original index
            
            if hasattr(obj, 'model_type') and obj.model_type in ["WallInner", "WallOuter", "Front", "Back"]:


                # Get the broken face indices
                broken = trimesh.repair.broken_faces(mesh)
                num_broken = len(broken)
                print(f"num_broken_faces:{num_broken}")
                #print(f"found {num_broken} broken faces for wall with id {obj.model_uid}.")

                            
                broken_faces = mesh.faces[broken]


                                # Get the corresponding vertex positions for the broken faces
                broken_vertices = mesh.vertices[broken_faces]

                # Print the broken faces indices and vertices
                print(f"Broken face indices: {broken}")
                print(f"Broken vertices coordinates: \n{broken_vertices}")


                def triangle_area(verts):
                    # verts: (3, 3) array
                    # Calculate two edge vectors
                    v0, v1, v2 = verts
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    # Area of triangle = 0.5 * norm of cross product of two edges
                    cross_prod = np.cross(edge1, edge2)
                    area = 0.5 * np.linalg.norm(cross_prod)
                    return area

                # Compute areas
                areas = np.array([triangle_area(face) for face in broken_vertices])

                # Total area
                total_broken_area = areas.sum()
                print(f"Total area of broken faces: {total_broken_area}")


                broken_vertices = mesh.vertices[broken_faces]
                flat_vertices = broken_vertices.reshape(-1, 3)

                tolerance = 1e-6  # For comparing floating-point numbers
                total_projected_area = 0

                # Check for single plane cases (YZ-parallel)
                if np.all(np.abs(flat_vertices[:, 0] - flat_vertices[0, 0]) < tolerance):
                    y_min = np.min(flat_vertices[:, 1])
                    y_max = np.max(flat_vertices[:, 1])
                    z_min = np.min(flat_vertices[:, 2])
                    z_max = np.max(flat_vertices[:, 2])
                    area_yz = (y_max - y_min) * (z_max - z_min)
                    total_projected_area = area_yz
                    print(f"All vertices on a plane parallel to YZ. Area: {area_yz}")

                # Check for single plane cases (XZ-parallel)
                elif np.all(np.abs(flat_vertices[:, 1] - flat_vertices[0, 1]) < tolerance):
                    x_min = np.min(flat_vertices[:, 0])
                    x_max = np.max(flat_vertices[:, 0])
                    z_min = np.min(flat_vertices[:, 2])
                    z_max = np.max(flat_vertices[:, 2])
                    area_xz = (x_max - x_min) * (z_max - z_min)
                    total_projected_area = area_xz
                    print(f"All vertices on a plane parallel to XZ. Area: {area_xz}")

                else:
                    # Handle multiple plane cases (based on constant x or y)
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

                    #TODO: find good threshold.
                    # i found a wall whhere even whena all faces were combined, it left a bit of thin vertical space.
                    # meaning we should not consider that as a gap for door and window and exclude. we need that area as threshold.
                    # or may be calculate the area of normal door and keep a value lower than that.
                    # it seems doors area is usually above 1.
                    area_tolerance=0.9
                    if total_projected_area>total_broken_area+area_tolerance: # add either conditions like if faces_broken=2 or less.
                        print("there is enough space available for either a wall or a window.")
                    else:
                        print("this is just a wall and no other elements can be fit into it.")

                
                #TODO: whaaaaaaaaaa, might need to adjust
                if num_broken>=6: #these are the  walls  that  contain doors and windows
                    # Get the corresponding vertex positions for the broken faces
                    broken_vertices = mesh.vertices[broken_faces]

                    # Print the broken faces indices and vertices
                    print(f"Broken face indices: {broken}")
                    print(f"Broken vertices coordinates: \n{broken_vertices}")



                    broken_vertices = mesh.vertices[broken_faces]
                    flat_vertices = broken_vertices.reshape(-1, 3)

                    tolerance = 1e-6  # For comparing floating-point numbers
                    total_projected_area = 0

                    # Step 2: Check which coordinate is constant (x, y, or z)
                    unique_x = np.unique(flat_vertices[:, 0])
                    unique_y = np.unique(flat_vertices[:, 1])
                    unique_z = np.unique(flat_vertices[:, 2])



                    threshold = 0.1  # You can adjust this threshold based on your needs

                    # Function to check if values are close enough
                    def is_close(values, threshold):
                        return np.ptp(values) < threshold  # ptp = max - min (range)

                    # Determine which coordinate is (almost) constant
                    if len(unique_x) == 1 or is_close(unique_x, threshold):
                        constant_coord = 'x'
                        constant_value = np.mean(flat_vertices[:, 0])  # take mean
                        remaining_coords = flat_vertices[:, 1:3]  # y and z
                    elif len(unique_y) == 1 or is_close(unique_y, threshold):
                        constant_coord = 'y'
                        constant_value = np.mean(flat_vertices[:, 1])  # take mean
                        remaining_coords = flat_vertices[:, [0, 2]]  # x and z
                    elif len(unique_z) == 1 or is_close(unique_z, threshold):
                        constant_coord = 'z'
                        constant_value = np.mean(flat_vertices[:, 2])  # take mean
                        remaining_coords = flat_vertices[:, [0, 1]]  # x and y
                    else:
                        #TODO: i dont  know what to do , i dont know what to do.
                        non_axis_aligned_wall_count += 1
                        print("this wall is not axis aligned.......")
                        continue
                                    




                            
                    
                            
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

                    # Loop over the first column values
                    for val in first_column:
                        if val < mean_first_column:
                            if (min_1 is None) or (val > min_1):  # Closest value less than mean
                                min_1 = val
                        elif val > mean_first_column:
                            if (max_1 is None) or (val < max_1):  # Closest value greater than mean
                                max_1 = val

                    print(f"Adjusted min_1 (less than mean): {min_1}")
                    print(f"Adjusted max_1 (greater than mean): {max_1}")

                    # squared_differences_first_column = (first_column - mean_first_column)**2
                    # distances_first_column = np.sqrt(squared_differences_first_column)
                    # # print(f"Distances of each point (first column) from the average:\n{distances_first_column}")

                    # # Get the indices of the two smallest distances in the first column
                    # smallest_distance_indices_first_column = np.argsort(distances_first_column)[:2]
                    # # print(f"Indices of the two smallest distances (first column): {smallest_distance_indices_first_column}")

                    # # Get the points (rows) corresponding to the two smallest distances in the first column
                    # point_min_distance_row = sorted_remaining_coords[smallest_distance_indices_first_column[0]]
                    # point_next_min_distance_row = sorted_remaining_coords[smallest_distance_indices_first_column[1]]
                    # # print(f"Point (row) with min distance (based on first column): {point_min_distance_row}")
                    # # print(f"Point (row) with next min distance (based on first column): {point_next_min_distance_row}")

                    # # Compare the first coordinates of these two points with the average of the first column
                    # if point_min_distance_row[0] < mean_first_column:
                    #     min_1 = point_min_distance_row[0]
                    #     max_1 = point_next_min_distance_row[0]
                    # else:
                    #     min_1 = point_next_min_distance_row[0]
                    #     max_1 = point_min_distance_row[0]

                    # print(f"min_1 (first column): {min_1}")
                    # print(f"max_1 (first column): {max_1}")

                

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
                            else: #windows (vs mac haha)
                                windows_meshes_to_concatenate.append(door_window_mesh)
                                print(f"wall id for windows (by IoU) is:{wall_id}, IoU: {iou_score}")
                                win_obj = Door(obj_index, wall_id, position_x, position_y, position_z, width, height, "window")
                                print(win_obj.to_language_string())
                                output_string += win_obj.to_language_string() + '\n' 
                        else:
                            print("no wall matching...")

                    else:
                        print("this wall also doesnt hap gap........................")
                else:
                    print(f"num of broken faces is:{num_broken}")
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

        else:
            # we are taking walls and dealing with them, anything apart from them in model_type. it is ignored.
            print(f"for {obj.model_type},Skipping language generation. (This is not wall.)")
          




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




    # -------------Visualization part ------------------
    scene_unique_id= scene_unique_id
    print(f"scene unique id: {scene_unique_id}")
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





    #furniture and wall test
    valid_furniture = filter_furniture_in_scene(wall_meshes_to_concatenate, furniture_meshes, enclosure_threshold=0.5)
    #furniture and furniture test
    furniture_furniture_collisions = filter_furniture_furniture_collisions(valid_furniture)
    if furniture_furniture_collisions:
        print("⚠️ Furniture-Furniture collisions detected.")
    else:
        print("✅ No undesired furniture-furniture collisions.")

    if valid_furniture and not furniture_furniture_collisions and non_axis_aligned_wall_count ==0:
        #----------------SAVING the script -----------------------------
        # to folder with desired filename and location.
        #TODO: change the output dir
        output_dir = "SCRIPTS"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name= f"{scene_unique_id}.txt"
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path,"w")as f:
            f.write(output_string)
        print(f"✅ succesfully wrote script to :{output_path}")
        valid_scene_ids.append(scene_unique_id) 
    else:
        print("❌ not as desired.")
        reasons = []
        if not valid_furniture:
            reasons.append("furniture-wall collision")
        if furniture_furniture_collisions:
            reasons.append("furniture-furniture collision")
        elif non_axis_aligned_wall_count >=1:
            reasons.append(f"{non_axis_aligned_wall_count} wall/s are not aligned to axis")

        #TODO: remove this after testing purupose
        output_dir = "SCRIPTS"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name= f"{scene_unique_id}.txt"
        output_path = os.path.join(output_dir, output_file_name)
        with open(output_path,"w")as f:
            f.write(output_string)
        print(f"✅ ❌ wrong but still wrote script to :{output_path}")
        valid_scene_ids.append(scene_unique_id) 


        reason_str = ", ".join(reasons)
        print(f"❌ {scene_unique_id} -> {reason_str}")
        invalid_scene_ids.append((scene_unique_id, reason_str))  




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










#                     #TODO: 
#                     def calculate_projected_area_broken_faces(mesh, tolerance=0.01):
#                         broken = trimesh.repair.broken_faces(mesh)
#                         num_broken = len(broken)
#                         print(f"num_broken_faces: {num_broken}")
#                         if num_broken == 0:
#                             return 0.0

#                         broken_faces = mesh.faces[broken]
#                         broken_vertices = mesh.vertices[broken_faces]
#                         flat_vertices = broken_vertices.reshape(-1, 3)
#                         print(f"Broken face indices: {broken}")
#                         print(f"Broken vertices coordinates: \n{broken_vertices}")

#                         total_projected_area = 0.0
#                         detected_planes = []

#                         def find_top_dominant(vertices, current_tolerance, top_n=2):
#                             dominant_info = []
#                             for i in [0, 1]:  # Check X and Y
#                                 unique_vals, counts = np.unique(np.round(vertices[:, i] / current_tolerance) * current_tolerance, return_counts=True)
#                                 for val, count in zip(unique_vals, counts):
#                                     if count > 0:
#                                         dominant_info.append({'axis': i, 'value': val, 'count': count})
#                             return sorted(dominant_info, key=lambda x: x['count'], reverse=True)[:top_n]

#                         top_dominants = find_top_dominant(flat_vertices, tolerance, top_n=2)

#                         processed_dominant_values = set()

#                         for dominant in top_dominants:
#                             axis = dominant['axis']
#                             value = dominant['value']

#                             if (axis, value) not in processed_dominant_values:
#                                 mask = np.isclose(flat_vertices[:, axis], value, atol=tolerance)
#                                 current_plane_vertices = flat_vertices[mask]

#                                 if len(current_plane_vertices) > 2:
#                                     if axis == 0:  # YZ-parallel
#                                         y_min = np.min(current_plane_vertices[:, 1])
#                                         y_max = np.max(current_plane_vertices[:, 1])
#                                         z_min = np.min(current_plane_vertices[:, 2])
#                                         z_max = np.max(current_plane_vertices[:, 2])
#                                         area = (y_max - y_min) * (z_max - z_min)
#                                         print(f"Found YZ-parallel plane (dominant X={value:.4f}). Area: {area:.4f}")
#                                         total_projected_area += area
#                                         detected_planes.append("yz")
#                                     elif axis == 1:  # XZ-parallel
#                                         x_min = np.min(current_plane_vertices[:, 0])
#                                         x_max = np.max(current_plane_vertices[:, 0])
#                                         z_min = np.min(current_plane_vertices[:, 2])
#                                         z_max = np.max(current_plane_vertices[:, 2])
#                                         area = (x_max - x_min) * (z_max - z_min)
#                                         print(f"Found XZ-parallel plane (dominant Y={value:.4f}). Area: {area:.4f}")
#                                         total_projected_area += area
#                                         detected_planes.append("xz")
#                                 processed_dominant_values.add((axis, value))

#                         unique_planes = sorted(list(set(detected_planes)))

#                         if not detected_planes:
#                             print("No dominant planar structures found.")
#                         elif len(unique_planes) == 2 and "xz" in unique_planes and "yz" in unique_planes:
#                             print("ok, there are two walls on different orthogonal planes (xz and yz).")
#                         elif len(unique_planes) == 1 and len(detected_planes) > 1:
#                             print(f"They are fragmented (multiple sections on the same plane: {unique_planes[0]}).")
#                         elif len(unique_planes) > 2:
#                             print(f"Unexpected case: more than two dominant plane orientations found: {', '.join(unique_planes)}")
#                         elif len(unique_planes) == 1 and len(detected_planes) == 1:
#                             print(f"Found a single, non-fragmented wall on the {unique_planes[0]} plane.")
#                         else:
#                             print("Undesired or ambiguous case.")

#                         return total_projected_area

#                     projected_area = calculate_projected_area_broken_faces(mesh)
#                     print(f"Total projected  wall area: {projected_area:.4f}")

