"""
a script to convert centralized dataformat to SceneScript format
"""

import os

# scene setting
import habitat_sim

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        sim_cfg.override_scene_light_defaults = settings[
            "override_scene_light_defaults"
        ]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
    color_sensor_1st_person_spec.uuid = "semantic"
    color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    color_sensor_1st_person_spec.resolution = [
        settings["height"],
        settings["width"],
    ]
    color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_1st_person_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_1st_person_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_default_settings(data_path):
    settings = {
        "width": 1280,  # Spatial resolution of the observations
        "height": 720,
        "scene_dataset": os.path.join(
            data_path, "hssd-hab/hssd-hab.scene_dataset_config.json"
        ),  # dataset path
        # "scene_dataset": os.path.join(
        #     data_path, "ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
        # ),  # dataset path
        # "scene": "NONE",  # Scene path
        # "scene": "/mnt/sv-share/habitat-sim/data/scene_datasets/ai2thor-hab/ai2thor-hab/configs/scenes/iTHOR/FloorPlan9_physics.scene_instance.json",  # Scene path
        "scene": os.path.join(data_path, "hssd-hab/scenes/102343992.scene_instance.json"),  # Scene path
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": 0.0,  # sensor pitch (x rotation in rads)
        "seed": 1,
        "enable_physics": False,  # enable dynamics simulation
        "print_semantic_scene": True,  # print semantic scene
        "load_semantic_mesh": True,
        # "semantic_mesh": os.path.join(
        #     data_path, "hssd-hab/hssd-hab.scene_dataset_config.json"
        # )
    }
    return settings

data_path = "/mnt/sv-share/habitat-sim/data/scene_datasets"
# data_path = "/mnt/sv-share/habitat-sim/data/scene_datasets/ai2thor-hab"
sim_settings = make_default_settings(data_path)
cfg = make_cfg(sim_settings)

sim = habitat_sim.Simulator(cfg)

print(sim)

# Managers of various Attributes templates
# # obj_attr_mgr = sim.get_object_template_manager()
# # obj_attr_mgr.load_configs(str(os.path.join(data_path, "objects/example_objects")))
# prim_attr_mgr = sim.get_asset_template_manager()
# stage_attr_mgr = sim.get_stage_template_manager()
# # Manager providing access to rigid objects
# rigid_obj_mgr = sim.get_rigid_object_manager()
# # get metadata_mediator
# metadata_mediator = sim.metadata_mediator
#
# # convert the data to SceneScript format

from habitat_sim.utils.common import quat_from_angle_axis

scene_handles = sim.metadata_mediator.get_scene_handles()
semantic_scene = sim.semantic_scene

print(semantic_scene.levels)
print(semantic_scene.objects)

