import trimesh
import numpy as np
from dataset_toolkits.layout.entity import Wall, Door, Window, Bbox
from dataset_toolkits.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform


def log(*args):
    print("=" * 10, *args, "=" * 10)


def find_closest_wall_by_iou(window_extra, all_extras, walls=["WallInner", "Front", "Back"]):
    window_mesh = trimesh.Trimesh(vertices=window_extra.xyz, faces=window_extra.faces)
    window_aabb = window_mesh.bounding_box.bounds
    window_min_xy = window_aabb[0][[0, 1]]
    window_max_xy = window_aabb[1][[0, 1]]

    max_iou = 0
    best_wall_id = None

    for _, wall_extra in enumerate([e for e in all_extras if e.model_type in walls]):
        wall_mesh = trimesh.Trimesh(vertices=wall_extra.xyz, faces=wall_extra.faces)
        wall_aabb = wall_mesh.bounding_box.bounds
        wall_min_xy = wall_aabb[0][[0, 1]]
        wall_max_xy = wall_aabb[1][[0, 1]]

        intersect_min = np.maximum(window_min_xy, wall_min_xy)
        intersect_max = np.minimum(window_max_xy, wall_max_xy)
        intersect_dims = np.maximum(intersect_max - intersect_min, 0)
        intersection_area = np.prod(intersect_dims)

        window_area = np.prod(window_max_xy - window_min_xy)
        wall_area = np.prod(wall_max_xy - wall_min_xy)
        union_area = window_area + wall_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0

        if iou > max_iou:
            max_iou = iou
            best_wall_id = wall_extra.model_uid

    return best_wall_id, max_iou


# Paths
path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
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
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

cat_objects = []

cat_objects = []
for s in d.scenes:
    for bbox in s.bboxes:
        cat_objects.append(bbox.label)
print(set(cat_objects))
exit(0)

for s in d.scenes:
    # collect cateogry of furniture


    if s.scene_id != "MasterBedroom-5863":
        continue

    log("Processing scene:", s.scene_id)
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}

    for i, extra in enumerate(s.extras):
        mesh = trimesh.Trimesh(vertices=extra.xyz, faces=extra.faces)
        mesh.apply_transform(transform)

        vertices = mesh.vertices
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        height = z_max - z_min

        if extra.model_type in ["WallInner", "Front", "Back"]: #  Ceiling
            # Find two most distant floor-level vertices in XY
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

            thickness = mesh.bounding_box_oriented.primitive.extents.min()
            thickness = max(thickness, 0.05)

            wall = Wall(i, ax, ay, az, bx, by, bz, height, thickness)
            print(wall.to_language_string())

        elif extra.model_type in ["Window", "Door"]:
            center = mesh.bounding_box.centroid
            position_x, position_y, position_z = center.tolist()
            width = mesh.bounding_box.extents[0] #TODO: validate for window
            height = mesh.bounding_box.extents[2]  # since Z is up

            best_wall_id, _ = find_closest_wall_by_iou(extra, s.extras, walls=["WallInner", "Front", "Back"])
            wall_id = extra2index.get(best_wall_id, 0)

            if extra.model_type == "Window":
                obj = Window(i, wall_id, position_x, position_y, position_z, width, height, "window")
            else:
                obj = Door(i, wall_id, position_x, position_y, position_z, width, height, "door")
            print(obj.to_language_string())

    for index, furniture in enumerate(s.bboxes):
        # 1. Load raw mesh
        mesh = furniture.raw_model()

        # 2. Apply model-to-scene transform
        vertices = furniture._transform(mesh.vertices)
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

        # 3. Apply Z-up transformation (if needed)
        mesh.apply_transform(transform)

        bbox = mesh.bounding_box_oriented
        center = mesh.centroid
        # center = bbox.centroid
        dims = bbox.extents  # dx, dy, dz

        # Extract heading (rotation around Z axis)
        # Compute rotation matrix from bounding box
        rot_matrix = bbox.primitive.transform[:3, :3]
        # Heading = angle between bbox local X-axis and world X-Y

        heading = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # yaw in radians

        class_name = furniture.label

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