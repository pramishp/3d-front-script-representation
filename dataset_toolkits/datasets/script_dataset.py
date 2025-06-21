import os
import glob
import random
import yaml
from torch.utils.data import Dataset, DataLoader
from dataset_toolkits.layout.layout import Layout

from model.llava import conversation as conversation_lib


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SceneRoomDataset(Dataset):
    """
    PyTorch Dataset for 3D-FRONT scripts. Returns dict with keys:
      - 'script': filtered script text (stage + randomized bboxes)
      - 'full_script': full script text
      - 'scene_room': identifier of the scene-room
    """

    def __init__(self, config, split='train'):
        self.root_dir = config['data']['root_dir']
        self.stage_cats = config['data'].get('stage_categories', ['wall', 'door', 'window'])
        self.include_bbox_prob = config['data'].get('include_bbox_prob', 0.0)
        self.train_split = config['data'].get('train_split', 0.8)
        self.random_seed = config['data'].get('random_seed', 42)
        self.split = split

        # 1) Gather all script files
        pattern = os.path.join(self.root_dir, '*__*.txt')
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f'No script files found with pattern {pattern}')

        # 2) Shuffle and split at file level
        random.seed(self.random_seed)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * self.train_split)
        # ensure at least 1 file in each split if possible
        if len(all_files) > 1:
            split_idx = max(1, min(split_idx, len(all_files) - 1))

        if split == 'train':
            self.files = all_files[:split_idx]
        else:  # 'val'
            self.files = all_files[split_idx:]

        if not self.files:
            raise RuntimeError(
                f"No script files for split={split!r} "
                f"(split_idx={split_idx}, total_files={len(all_files)})"
            )

        print(f"[SceneRoomDataset] split={split}: "
              f"{len(self.files)}/{len(all_files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            txt = f.read().strip()
        # Parse using Layout
        scene = Layout(txt)

        # do normalization and discretization
        scene.normalize_and_discretize()

        # Full script as GT
        full_script = scene.to_language_string()

        # Stage elements
        stage_lines = []
        for cat in self.stage_cats:
            attr = cat + 's'  # e.g., 'wall' -> 'walls'
            if hasattr(scene, attr):
                entities = getattr(scene, attr)
                for ent in entities:
                    stage_lines.append(ent.to_language_string())
            else:
                raise ValueError(f"Unknown stage category '{cat}' for Layout")

        # Bbox elements
        bbox_lines = [bbox.to_language_string() for bbox in scene.bboxes]

        # Randomize bboxes
        kept_bboxes = []
        predicted_bboxes = []
        for b in bbox_lines:
            if random.random() < self.include_bbox_prob:
                kept_bboxes.append(b)
            else:
                predicted_bboxes.append(b)

        # Final script
        script_lines = stage_lines + kept_bboxes
        script = "\n".join(script_lines)

        question = script
        answer = "\n".join(predicted_bboxes)
        SYSTEM_MSG = ""

        # build conversation
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)

        return {
            # 'script': script,
            # 'full_script': full_script,
            'scene_room': os.path.basename(file_path).replace('.txt', ''),
            "conversation": conv.get_prompt(),
            "question": question,
            "answer": answer
        }


def build_dataloader(config_path, split='train'):
    """
    Utility to build DataLoader from YAML config.
    """
    config = load_config(config_path)
    dataset = SceneRoomDataset(config, split)
    dl_params = config['dataloader']

    return DataLoader(
        dataset,
        **dl_params
    )


# Example usage
if __name__ == '__main__':
    train_loader = build_dataloader('dataset_toolkits/cfgs/dataset_train_debug.yml', 'train')
    # val_loader = build_dataloader('config.yml', 'val')
    for batch in train_loader:
        print(batch['scene_room'][0], "\n", batch['script'][0])




