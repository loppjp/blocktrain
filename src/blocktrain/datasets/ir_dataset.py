import json
from pathlib import Path

import torch

from blocktrain.factories.loader import load


def ir_dataset_factory(
    top_level_dataset_folder: Path,
    indicies_json_file_path: Path,
    child_dataset_folder_path_glob: str = "*_1",
    child_spec: dict = {
        "__module__": "blocktrain.datasets.ir_ram_dataset",
        "__class__":  "FolderImageRAMDataset"
    }
) -> torch.utils.data.Dataset:
    """
    Creates a concat dataset based off of child dataset spec
    """
    dataset_folders: list[Path] = list(top_level_dataset_folder.glob(child_dataset_folder_path_glob))

    full_dataset: torch.utils.data.Dataset = torch.utils.data.ConcatDataset([
        load(
            list(dataset_folder.glob("*.json"))[0], # annotations file
            dataset_folder,
            **child_spec
        )
        for dataset_folder in dataset_folders
    ])

    return torch.utils.data.Subset(
        full_dataset,
        json.load(indicies_json_file_path.open())
    )
