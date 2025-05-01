from dataclasses import dataclass, field
import yaml
import os
from typing import Union, Dict


def load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


@dataclass
class Configs:
    # This dictionary will either contain further Configs objects or the actual data from YAML files
    contents: Dict[str, Union["Configs", dict]] = field(default_factory=dict)

    def __getattr__(self, item):
        try:
            return self.contents[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")


def load_configs(directory: str) -> Configs:
    configs = Configs()
    for root, dirs, files in os.walk(directory):
        # Get the path relative to the starting directory
        rel_path = os.path.relpath(root, directory)
        path_parts = rel_path.split(os.sep) if rel_path != "." else []
        current = configs
        # Traverse or create the necessary nested Configs objects
        for part in path_parts:
            if part not in current.contents:
                current.contents[part] = Configs()
            current = current.contents[part]

        # Load YAML files and assign to current configs
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filename_without_ext = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                current.contents[filename_without_ext] = load_yaml_file(full_path)

    return configs


# Usage
root_directory = "configs"
CONFIGS = load_configs(root_directory)
