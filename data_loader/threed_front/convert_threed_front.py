
import trimesh
import numpy as np

from data_loader.layout.entity import Wall, Door, Window, Bbox

from data_loader.threed_front.threed_front import ThreedFront


def log(*args):
    """Log messages to the console."""
    print("="*10, *args, "="*10)


def find_closest_wall_by_iou(window_extra, all_extras):
    """
    Match a window to a wall by 2D IoU (projected onto XY plane).
    This works when walls are aligned along Z (i.e., Z is constant).
    """
    window_mesh = trimesh.Trimesh(vertices=window_extra.xyz, faces=window_extra.faces)
    window_aabb = window_mesh.bounding_box.bounds
    window_min_xy = window_aabb[0][[0, 1]]
    window_max_xy = window_aabb[1][[0, 1]]

    max_iou = 0
    best_wall_id = None

    for _, wall_extra in enumerate([e for e in all_extras if "Wall" in e.model_type]):
        wall_mesh = trimesh.Trimesh(vertices=wall_extra.xyz, faces=wall_extra.faces)
        wall_aabb = wall_mesh.bounding_box.bounds
        wall_min_xy = wall_aabb[0][[0, 1]]
        wall_max_xy = wall_aabb[1][[0, 1]]

        # 2D AABB intersection (XY plane)
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

path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
path_to_model_info = "/mnt/sv-share/3DFRONT/3D-FUTURE-model/model_info.json"
path_to_3d_future_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FUTURE-model"

# scene_id = MasterBedroom-5863

d = ThreedFront.from_dataset_directory(
    path_to_3d_front_dataset_directory,
    path_to_model_info,
    path_to_3d_future_dataset_directory,
    path_to_room_masks_dir=None,
    path_to_bounds=None,
    filter_fn=lambda s: s
)

print(d)


for s in d.scenes:
    if s.scene_id != "MasterBedroom-5863":
        continue
    log("Processing scene: ", s.scene_id,)
    extra2index = {key.model_uid: val for val, key in enumerate(s.extras)}
    # read walls, floor, ceiling, windows, doors
    for index, extra in enumerate(s.extras):
        # vertices = extra.xyz - s.floor_plan_centroid
        vertices = extra.xyz
        mesh = trimesh.Trimesh(vertices=vertices, faces=extra.faces)
        transform = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        # Axis-aligned bounding box
        aabb = mesh.bounding_box.bounds  # shape: (2, 3)
        height = aabb[1, 1] - aabb[0, 1]  # y-axis height
        width = aabb[1, 0] - aabb[0, 0]
        thickness = mesh.bounding_box_oriented.primitive.extents.min()
        # mesh.apply_transform(transform)
        center = mesh.bounding_box.centroid
        position_x, position_y, position_z = center.tolist()
        if extra.model_type in ["WallInner"]: #  "",
            # process as walls
            # id: int
            # ax: float
            # ay: float
            # az: float
            # bx: float
            # by: float
            # bz: float
            # height: float
            # thickness: float
            # entity_label: str = "wall"


            mesh = trimesh.Trimesh(vertices=extra.xyz, faces=extra.faces)
            # mesh.apply_transform(transform)

            vertices = mesh.vertices
            floor_y = vertices[:, 1].min()
            floor_vertices = vertices[np.isclose(vertices[:, 1], floor_y, atol=1e-3)]
            floor_vertices = np.unique(floor_vertices, axis=0)

            from scipy.spatial.distance import pdist, squareform

            xz = floor_vertices[:, [0, 2]]
            distances = squareform(pdist(xz))
            i, j = np.unravel_index(np.argmax(distances), distances.shape)

            start = floor_vertices[i]
            end = floor_vertices[j]
            ax, ay, az = start
            bx, by, bz = end
            z_min = mesh.vertices[:, 2].min()
            z_max = mesh.vertices[:, 2].max()
            height = z_max - z_min
            thickness = mesh.bounding_box_oriented.primitive.extents.min()
            thickness = max(thickness, 0.01)

            wall = Wall(index, ax, ay, az, bx, by, bz, height, thickness)
            print(wall.to_language_string())

        elif extra.model_type in ["Floor"]:
            # skip floor
            pass
        elif extra.model_type in ["Window"]:
            # id: int
            # wall_id: int
            # position_x: float
            # position_y: float
            # position_z: float
            # width: float
            # height: float
            # entity_label: str = "door"

            #TODO: to find wall_id, run IOU on the wall and window meshes, the one with the highest IOU is the wall

            best_wall_id, max_iou = find_closest_wall_by_iou(extra, s.extras)

            if best_wall_id:
                window = Window(i, extra2index[best_wall_id], position_x, position_y, position_z, width, height, "window")
            else:
                window = Window(i, 000, position_x, position_y, position_z, width, height, "window")
            print(window.to_language_string())
            pass
        elif extra.model_type in ["Door"]:
            # process as door
            # id: int
            # wall_id: int
            # position_x: float
            # position_y: float
            # position_z: float
            # width: float
            # height: float
            # entity_label: str = "door"

            best_wall_id, max_iou = find_closest_wall_by_iou(extra, s.extras)

            if best_wall_id:
                door = Door(i, extra2index[best_wall_id], position_x, position_y, position_z, width, height, "door")
            else:
                door = Door(i, 000, position_x, position_y, position_z, width, height, "door")
            print(door.to_language_string())

    print()
    break
    # read objects
    for furniture in s.bboxes:
        pass


