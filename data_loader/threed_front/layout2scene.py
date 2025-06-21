import sys
import os

lisa_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(lisa_path)
print("Python path updated:", sys.path) # Debugging line


import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from data_loader.layout.layout import Layout
from scipy.spatial.transform import Rotation as R

# Paths
path_to_3d_front_dataset_directory = "/home/ajad/Desktop/codes/LISA"

# path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
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

# output_dir ="more_tests..."
# os.makedirs(output_dir, exist_ok=True)


def log(*args):
    print("=" * 10, *args, "=" * 10)


# meshes={}
# def log(*args):
#     print("=" * 10, *args, "=" * 10)
# first_three_scenes_data = d.scenes[:1]
# for s in first_three_scenes_data:
#     log("Processing scene:", s.scene_id)
#     print(f" furnitures in the scene are: \n")
#     for index, furniture in enumerate(s.bboxes):
#         try:
#             mesh = furniture.raw_model()

#             if mesh is not None:
#                 name=furniture.label
#                 meshes[name] = {
#                         "faces": mesh.faces.tolist(),
#                         "vertices": mesh.vertices.tolist()
#                     }
#                 print(f" {name}")
#             else:
#                 print("furniture has no mesh.")
#         except Exception as e:
#             print(f"error during furniture mesh iteration: {e}")

#     print(f"meshes dict: \n {meshes}")




import tqdm

all_furniture_labels = {
    "three-seat/multi-person sofa",
    "bed frame",
    "tv stand",
    "loveseat sofa",
    "king-size bed",
    "drawer chest/corner cabinet",
    "bunk bed",
    "ceiling lamp",
    "tea table",
    "wine cabinet",
    "barstool",
    "corner/side table",
    "coffee table",
    "wine cooler",
    "bar",
    "shoe cabinet",
    "u-shaped sofa",
    "floor lamp",
    "hanging chair",
    "double bed",
    "sideboard/side cabinet/console",
    "chaise longue sofa",
    "shelf",
    "two-seat sofa",
    "dining chair",
    "single bed",
    "armchair",
    "footstool/sofastool/bed end stool/stool",
    "three-seat/multi-seat sofa",
    "dressing table",
    "unknown_category",
    "pendant lamp",
    "wall lamp",
    "lounge chair/cafe chair/office chair",
    "folding chair",
    "couch bed",
    "kids bed",
    "nightstand",
    "dining table",
    "bookcase/jewelry armoire",
    "l-shaped sofa",
    "sideboard/side cabinet/console table",
    "children cabinet",
    "dressing chair",
    "wardrobe",
    "classic chinese chair",
    "lounge chair/book-chair/computer chair",
    "lazy sofa",
    "desk",
    "round end table",
}

meshes = {}


num_total_furniture = len(all_furniture_labels)
progress_bar = tqdm.tqdm(total=num_total_furniture, desc="Furniture Meshes Collected")

first_hundred_scenes= d.scenes



for s in first_hundred_scenes:
    log("Processing scene:", s.scene_id)
    print(f" furnitures in the scene are: \n")
    for index, furniture in enumerate(s.bboxes):
        name = furniture.label
        scale = furniture.scale
        print(f" {name} — Scale: {scale}")
        if name not in ['dining table']:
            if name not in meshes:
                try:
                    mesh = furniture.raw_model()  #raw_model_transformed()
                    #/mnt/sv-share/3DFRONT/3D-FUTURE-model/10821dd8-229a-480c-b6fd-68b4d7d20bc0  --> this contains nightstand future model 
                    if mesh is not None:
                        meshes[name] = {
                            "faces": mesh.faces.tolist(),
                            "vertices": mesh.vertices.tolist(),
                        }
                        print(f" Added mesh for: {name}")
                        progress_bar.update(1)
                        print(f" Total unique furniture meshes: {len(meshes)}")
                        if len(meshes) == num_total_furniture:
                            print("\nCompleted. Meshes for all furniture labels obtained.")
                            print(f"Final meshes dictionary: \n {meshes}")
                            progress_bar.close()
                            exit()
                    else:
                        print(f"Furniture '{name}' has no mesh.")
                except Exception as e:
                    print(f"Error during furniture mesh iteration for '{name}': {e}")
            else:
                print(f"Skipping '{name}': Mesh already exists.")
    print(f"Current meshes dict: \n {meshes.keys()}")

progress_bar.close()
print("\nFinished processing all scenes, but not all furniture labels might have been obtained.")
# print(f"Final meshes dictionary: \n {meshes}")





#now, the meshes dictionary has what we need. 
# now, we parse the pred.txt file that we have.




import os
pred_txt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pred.txt')
try:
    with open(pred_txt_path, 'r') as f:
        content = f.read()
        print(f"Contents of pred.txt:\n{content}")
except FileNotFoundError:
    print(f"Error: File not found at {pred_txt_path}")



    
# layout = Layout(content)
# print("Walls:")
# for wall in layout.walls:
#     print(wall)

# print("\nDoors:")
# for door in layout.doors:
#     print(door)

# print("\nWindows:")
# for window in layout.windows:
#     print(window)

# print("\nBounding Boxes:")
# for bbox in layout.bboxes:
#     print(bbox)
  
layout = Layout(content)
print(layout)
list_of_mesh_to_concatenate=[]



# for bbox in layout.bboxes:
#     class_name = bbox.class_name
#     print(f"Found bbox with class name: {class_name}")

#     if class_name in meshes:
#         print(f" corresponding mesh found for class name: {class_name}")

#         mesh_data = meshes[class_name]
#         vertices = np.array(mesh_data['vertices'])

#         faces = np.array(mesh_data['faces'])
#         mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

#         mesh.apply_transform(transform)
#         list_of_mesh_to_concatenate.append(mesh)

#     else:
#         print(f"No corresponding mesh found for class name: {class_name}")
#     print("-" * 20)






import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

for bbox in layout.bboxes:
    class_name = bbox.class_name
    scale = np.array([bbox.scale_x, bbox.scale_y, bbox.scale_z])
    position = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
    angle = bbox.angle_z  # in radians

    print(f"Found bbox with class name: {class_name} — Scale: {scale}, Position: {position}, Angle: {angle}")

    if class_name in meshes:
        print(f"  mesh found for class name: {class_name}")
        mesh_data = meshes[class_name]

        vertices = np.array(mesh_data['vertices'])
        faces = np.array(mesh_data['faces'])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 1. Scale

        
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(scale)

        # 2. Rotation (around z axis)
        rot_matrix = np.eye(4)
        # rot_matrix[:3, :3] = R.from_rotvec([0, 0, angle]).as_matrix()


        # First, rotate 90 degrees around the Y-axis
        rot_matrix[:3, :3] = R.from_rotvec([0, np.pi / 2, 0]).as_matrix()

        # Then rotate 90 degrees anticlockwise around the X-axis
        rot_matrix[:3, :3] = R.from_rotvec([np.pi / 2, 0, 0]).as_matrix() @ rot_matrix[:3, :3]


        # Now apply an additional 180-degree rotation around the Z-axisp
        rot_z_180 = R.from_rotvec([0, 0, -np.pi/2]).as_matrix()
        rot_matrix[:3, :3] = rot_z_180 @ rot_matrix[:3, :3]

        # 3. Translation
        trans_matrix = np.eye(4)
        trans_matrix[:3, 3] = position



                # Standardize mesh to unit box if not already
        mesh.apply_translation(-mesh.centroid)

        original_extents = mesh.bounding_box.extents
        epsilon = 1e-6
        scale_to_unit = 1.0 / (original_extents + epsilon)
        mesh.apply_scale(scale_to_unit)

        # Combine: T * R * S
        transform2 = trans_matrix @ rot_matrix @ scale_matrix

        # Apply transformation
        mesh.apply_transform(transform2)
        # mesh.apply_transform(transform)
        list_of_mesh_to_concatenate.append(mesh)
    else:
        print(f"No mesh found for class name: {class_name}")












if list_of_mesh_to_concatenate:
    concatenated_mesh = trimesh.util.concatenate(list_of_mesh_to_concatenate)
    output_filename = "layout2scene_mesh.obj"
    concatenated_mesh.export(output_filename)
    print(f"succesffuly concatenated and exported to {output_filename}")
else:
    print("no valid furnitures meshes to concatenate")






# for s in d.scenes: 
#     log("Processing scene:", s.scene_id)
#     output_string =""
#     list_of_mesh_to_concatenate=[]

#     for index, furniture in enumerate(s.bboxes):
#         # 1. Load raw mesh
#         mesh = furniture.raw_model()
#         if mesh is not None:

#             class_name = furniture.label
#             # 2. Apply model-to-scene transform
#             vertices = furniture._transform(mesh.vertices)
#             mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

#             # 3. Apply Z-up transformation (if needed) (needed i guess, else the furnitures are outside the layout formed from extras!)
#             mesh.apply_transform(transform)

#             list_of_mesh_to_concatenate.append(mesh)

#             bbox = mesh.bounding_box
#             #center = mesh.centroid
#             center = bbox.primitive.transform[:3,3]
#             # center = bbox.centroid
#             dims = bbox.extents  # dx, dy, dz

#             # Extract heading (rotation around Z axis)
#             # Compute rotation matrix from bounding box
#             rot_matrix = bbox.primitive.transform[:3, :3]
#             # Heading = angle between bbox local X-axis and world X-Y

#             heading = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # yaw in radians

#             # id: int
#             # class_name: str
#             # position_x: float            # 2. Apply model-to-scene transform
#             vertices = furniture._transform(mesh.vertices)
#             # position_y: float
#             # position_z: float
#             # angle_z: float
#             # scale_x: float
    
    
    
#             # scale_y: float
#             # scale_z: float
#             # entity_label: str = "bbox"
#             bbox = Bbox(index, class_name, center[0], center[1], center[2], heading, dims[0], dims[1], dims[2])
#             print(bbox.to_language_string())
  
#             output_string +=bbox.to_language_string()+'\n'

#     if list_of_mesh_to_concatenate:
#         concatenated_mesh = trimesh.util.concatenate(list_of_mesh_to_concatenate)
#         output_filename = "concatenated_mesh.obj"
#         concatenated_mesh.export(output_filename)
#         #print(f"succesffuly concatenated and exported to {output_filename}")
#     else:
#         print("no valid furnitures meshes to concatenate")

   


#    #load around 20 scenes, get all the furnitures objects in them.
#    #for each furniture, store differently, but store what
#    #store raw model and classname as furntiure.label
#    #meshes = {}
#     # name = furniture.label 
#     # meshes[name] = mesh
#    # for all 50 classes, store on furniture for each.




#    #then, load txt file
#    #parse it.
#    #then for bbox=(fdlf)
#    #if that, then load correspoinding mesh
#    #transform it.
#    #

