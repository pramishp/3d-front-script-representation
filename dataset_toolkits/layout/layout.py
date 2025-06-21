import numpy as np
from scipy.spatial.transform import Rotation as R
from dataset_toolkits.layout.entity import Wall, Door, Window, Bbox, NORMALIZATION_PRESET

furniture_priority = {
    'bed': 20,
    'king-size bed': 20,
    'double bed': 20,
    'single bed': 19,
    'kids bed': 19,
    'bunk bed': 19,
    'u-shaped sofa': 20,
    'l-shaped sofa': 20,
    'three-seat/multi-seat sofa': 19,
    'three-seat/multi-person sofa': 19,
    'dining table': 20,
    'desk': 18,
    'couch bed': 18,
    'wardrobe': 18,
    'lazy sofa': 15,
    'bed frame': 17,

    'drawer chest/corner cabinet': 17,
    'tv stand': 17,
    'two-seat sofa': 14,
    'loveseat sofa': 14,
    'chaise longue sofa': 13,
    'armchair': 13,
    'lounge chair/book-chair/computer chair': 13,
    'dressing table': 12,
    'nightstand': 12,
    'shoe cabinet': 12,
    'shelf': 12,
    'bar': 12,
    'barstool': 11,
    'dining chair': 10,
    'dressing chair': 10,
    'corner/side table': 10,
    'coffee table': 10,
    'tea table': 10,
    'floor lamp': 9,
    'ceiling lamp': 9,
    'pendant lamp': 9,
    'wall lamp': 8,
    'wine cabinet': 8,
    'wine cooler': 8,
    'bookcase/jewelry armoire': 8,
    'footstool/sofastool/bed end stool/stool': 7,
    'hanging chair': 7,
    'round end table': 7,
    'classic chinese chair': 6,
    'folding chair': 6,
    'lounge chair/cafe chair/office chair': 6,
    'children cabinet': 6,
    'sideboard/side cabinet/console': 5,
    'sideboard/side cabinet/console table': 5,
    'unknown_category': 0
}


class Layout:
    def __init__(self, str: str = None):
        self.walls = []
        self.doors = []
        self.windows = []
        self.bboxes = []

        if str:
            self.from_str(str)

    @staticmethod
    def get_grid_size():
        world_min, world_max = NORMALIZATION_PRESET["world"]
        return (world_max - world_min) / NORMALIZATION_PRESET["num_bins"]

    @staticmethod
    def get_num_bins():
        return NORMALIZATION_PRESET["num_bins"]

    def from_str(self, s: str):
        s = s.lstrip("\n")
        lines = s.split("\n")
        # wall lookup table
        existing_walls = []
        for line in lines:
            try:
                label = line.split("=")[0]
                entity_id = int(label.split("_")[1])
                entity_label = label.split("_")[0]

                # extract params
                start_pos = line.find("(")
                end_pos = line.find(")")
                params = line[start_pos + 1 : end_pos].split(",")

                if entity_label == Wall.entity_label:
                    wall_args = [
                        "ax",
                        "ay",
                        "az",
                        "bx",
                        "by",
                        "bz",
                        "height",
                        "thickness",
                    ]
                    wall_params = dict(zip(wall_args, params[0:8]))
                    entity = Wall(id=entity_id, **wall_params)
                    existing_walls.append(entity_id)
                    self.walls.append(entity)
                elif entity_label == Door.entity_label:
                    wall_id = int(params[0].split("_")[1])
                    if wall_id not in existing_walls:
                        continue

                    door_args = [
                        "position_x",
                        "position_y",
                        "position_z",
                        "width",
                        "height",
                    ]
                    door_params = dict(zip(door_args, params[1:6]))
                    entity = Door(
                        id=entity_id,
                        wall_id=wall_id,
                        **door_params,
                    )
                    self.doors.append(entity)
                elif entity_label == Window.entity_label:
                    wall_id = int(params[0].split("_")[1])
                    if wall_id not in existing_walls:
                        continue

                    window_args = [
                        "position_x",
                        "position_y",
                        "position_z",
                        "width",
                        "height",
                    ]
                    window_params = dict(zip(window_args, params[1:6]))
                    entity = Window(
                        id=entity_id,
                        wall_id=wall_id,
                        **window_params,
                    )
                    self.windows.append(entity)
                elif entity_label == Bbox.entity_label:
                    class_name = params[0]
                    bbox_args = [
                        "position_x",
                        "position_y",
                        "position_z",
                        "angle_z",
                        "scale_x",
                        "scale_y",
                        "scale_z",
                    ]
                    bbox_params = dict(zip(bbox_args, params[1:8]))
                    entity = Bbox(
                        id=entity_id,
                        class_name=class_name,
                        **bbox_params,
                    )
                    self.bboxes.append(entity)
            except Exception as e:
                continue

    def to_boxes(self):
        boxes = []
        lookup = {}
        for wall in self.walls:
            # assume the walls has a thickness of 0.0 for now
            thickness = 0.0
            corner_a = np.array([wall.ax, wall.ay, wall.az])
            corner_b = np.array([wall.bx, wall.by, wall.bz])
            length = np.linalg.norm(corner_a - corner_b)
            direction = corner_b - corner_a
            angle = np.arctan2(direction[1], direction[0])
            lookup[wall.id] = {"wall": wall, "angle": angle}

            center = (corner_a + corner_b) * 0.5 + np.array([0, 0, 0.5 * wall.height])
            scale = np.array([length, thickness, wall.height])
            rotation = R.from_rotvec([0, 0, angle]).as_matrix()
            box = {
                "id": wall.id,
                "class": Wall.entity_label,
                "label": Wall.entity_label,
                "center": center,
                "rotation": rotation,
                "scale": scale,
            }
            boxes.append(box)

        for fixture in self.doors + self.windows:
            wall_id = fixture.wall_id
            wall_info = lookup.get(wall_id, None)
            if wall_info is None:
                continue

            wall = wall_info["wall"]
            angle = wall_info["angle"]
            thickness = wall.thickness

            center = np.array(
                [fixture.position_x, fixture.position_y, fixture.position_z]
            )
            scale = np.array([fixture.width, thickness, fixture.height])
            rotation = R.from_rotvec([0, 0, angle]).as_matrix()
            class_prefix = 1000 if fixture.entity_label == Door.entity_label else 2000
            box = {
                "id": fixture.id + class_prefix,
                "class": fixture.entity_label,
                "label": fixture.entity_label,
                "center": center,
                "rotation": rotation,
                "scale": scale,
            }
            boxes.append(box)

        for bbox in self.bboxes:
            center = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
            scale = np.array([bbox.scale_x, bbox.scale_y, bbox.scale_z])
            rotation = R.from_rotvec([0, 0, bbox.angle_z]).as_matrix()
            class_name = bbox.class_name
            box = {
                "id": bbox.id + 3000,
                "class": Bbox.entity_label,
                "label": class_name,
                "center": center,
                "rotation": rotation,
                "scale": scale,
            }
            boxes.append(box)

        return boxes

    def get_entities(self):
        return self.walls + self.doors + self.windows + self.bboxes

    def normalize_and_discretize(self):
        for entity in self.get_entities():
            entity.normalize_and_discretize()

    def undiscretize_and_unnormalize(self):
        for entity in self.get_entities():
            entity.undiscretize_and_unnormalize()

    def translate(self, translation: np.ndarray):
        for entity in self.get_entities():
            entity.translate(translation)

    def rotate(self, angle: float):
        for entity in self.get_entities():
            entity.rotate(angle)

    def scale(self, scale: float):
        for entity in self.get_entities():
            entity.scale(scale)

    def to_language_string(self):
        entity_strings = []
        # Walls, doors, windows are prioritized uniformly and added first
        all_structural = self.walls + self.doors + self.windows
        for entity in all_structural:
            entity_strings.append(entity.to_language_string())

        # Sort Bboxes based on priority weight (descending)
        sorted_bboxes = sorted(
            self.bboxes,
            key=lambda x: furniture_priority.get(x.class_name, 5),
            reverse=True
        )

        for bbox in sorted_bboxes:
            entity_strings.append(bbox.to_language_string())

        return "\n".join(entity_strings)


if __name__ == "__main__":

    file = '../tmp/scene.txt'
    # read file
    with open(file, "r") as f:
        txt = f.read()

    scene = Layout(txt)

    print(scene.to_language_string())
