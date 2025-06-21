import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from itertools import combinations


def get_colliding_meshes(meshes, threshold_fraction=0.05, min_count=10):
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



output_dir ="more_tests..."
os.makedirs(output_dir, exist_ok=True)


def log(*args):
    print("=" * 10, *args, "=" * 10)


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
path_to_3d_front_dataset_directory = "/home/ajad/Desktop/codes/LISA"

#path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
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

for s in d.scenes: 
    if s.scene_id != "MasterBedroom-5863": #OtherRoom-32459 #Hallway-44736 #LivingDiningRoom-2042 #SecondBedroom-17992 #DiningRoom-4603 #MasterBedroom-5863  #MasterBedroom-6776 #SecondBedroom-282617 SecondBedroom-12951
        continue #Library-14893 #SecondBedroom-6353 #SecondBedroom-7177 LivingDiningRoom-3474

    log("Processing scene:", s.scene_id)
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}

    output_string =""
   
    windows_meshes_to_concatenate =[] # to visualize windows_meshes with blue color, but here the walls are also added to it.
    wall_meshes=[] # contains all meshes of walls, to check intersection with all furnitures
    furniture_meshes =[] #contains all furnitures meshes to compare intersection with walls


    for i, extra in enumerate(s.extras):
        mesh = trimesh.Trimesh(vertices=extra.xyz, faces=extra.faces)
        mesh.apply_transform(transform)

        vertices = mesh.vertices
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        height = z_max - z_min

        if extra.model_type in ["WallInner", "Front", "Back",]: # "WallOuter","WallBottom","WallTop"
            windows_meshes_to_concatenate.append(mesh)
            # Find two most distant floor-level vertices in XY
            wall_meshes.append(mesh)
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

            wall = Wall(i, ax, ay, az, bx, by, bz, height, thickness)
            # print(f"wall_id:{i}-model_uid:{extra.model_uid}")
            print(wall.to_language_string())
            output_string +=wall.to_language_string()+'\n'

        
        elif extra.model_type in ["Window", "Door","interiorDoor","entryDoor"]: #interiorDoor, entryDoor, outdoor,
            bbox=mesh.bounding_box
            min_corner, max_corner = mesh.bounds
            win_pos_x, win_pos_y, win_pos__z =((min_corner+max_corner)/2).tolist()
            win_width =max(max_corner[1]-min_corner[1], max_corner[0]-min_corner[0])
            win_height =max_corner[2]-min_corner[2]
            center = bbox.centroid
            dims=bbox.extents 
            rot_matrix =bbox.primitive.transform[:3,:3]
            position_x, position_y, position_z = center.tolist()
            width = np.max(dims[:2]) #todo
            height = dims[2]  # since Z is up

            best_wall_id, _ = find_closest_wall_by_iou(extra, s.extras, walls=["WallInner","WallOuter","WallTop","WallBottom" "Front", "Back"])
            wall_id = extra2index.get(best_wall_id, 0)

            if extra.model_type == "Window":
                windows_meshes_to_concatenate.append(mesh)
                # print(f"wall id for windows is:{wall_id}")
                obj = Window(i, wall_id, win_pos_x, win_pos_y, win_pos__z, win_width, win_height, "window")
            else:
                # print(f"wall id for door is : {wall_id}")
                obj = Door(i, wall_id, position_x, position_y, position_z, width, height, "door")
            print(obj.to_language_string())
            output_string +=obj.to_language_string()+'\n'

        


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

    

