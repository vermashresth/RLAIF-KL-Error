import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import wandb
from huggingface_hub import HfApi, get_collection, RepoCard, metadata_update
from huggingface_hub.utils import (
    RevisionNotFoundError,
    EntryNotFoundError,
    HfHubHTTPError,
)
from transformers import HfArgumentParser
from datasets import get_dataset_config_names
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DataFilesNotFoundError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(token=HUGGINGFACE_CONFIGS["token"])
WANDB_CONFIGS = CONFIGS.services.wandb
WANDBAPI = wandb.Api(api_key=WANDB_CONFIGS["api_key"])


@dataclass
class ScriptArguments:
    tag: Optional[str] = field(
        default=None,
        metadata={"help": "tag for the experiment, to be deleted"},
    )
    clear_models: bool = field(
        default=True,
        metadata={"help": "delete all huggingface models with the tag"},
    )
    clear_evaluations: bool = field(
        default=True,
        metadata={"help": "delete all huggingface evaluation with the tag"},
    )
    clear_wandb: bool = field(
        default=True,
        metadata={"help": "delete all wandb runs with the tag"},
    )
    clear_datasets: bool = field(
        default=False,
        metadata={"help": "delete all preprocessed datasets"},
    )
    clear_all_hf: bool = field(
        default=False,
        metadata={
            "help": "delete all huggingface repos, warning: do not use unless necessary"
        },
    )


def delete_tagged_huggingface_models(tag):
    collection = get_collection(HUGGINGFACE_CONFIGS["collections"]["models"])
    for item in collection.items:
        tags = [branch.name for branch in HFAPI.list_repo_refs(item.item_id).branches]

        if len(set(tags) - set(["main", tag])) == 0:
            HFAPI.delete_repo(
                repo_id=item.item_id,
                repo_type="model",
            )
        else:
            try:
                HFAPI.delete_branch(
                    repo_id=item.item_id,
                    branch=tag,
                    repo_type="model",
                )
            except RevisionNotFoundError:
                pass


def delete_tagged_huggingface_evaluations(tag):
    collection = get_collection(HUGGINGFACE_CONFIGS["collections"]["evaluations"])
    for item in collection.items:
        try:
            tags = get_dataset_config_names(item.item_id)
        except EmptyDatasetError:
            tags = []
        except DataFilesNotFoundError:
            tags = []
        except HfHubHTTPError:
            continue

        if len(set(tags) - set([tag])) == 0:
            HFAPI.delete_repo(
                repo_id=item.item_id,
                repo_type="dataset",
            )
        else:
            try:
                HFAPI.delete_folder(
                    path_in_repo=tag + "/",
                    repo_id=item.item_id,
                    repo_type="dataset",
                )
                card = RepoCard.load(
                    item.item_id,
                    repo_type="dataset",
                )
                org_metadata = card.data.to_dict()
                new_metadata = {
                    "dataset_info": [
                        d
                        for d in org_metadata["dataset_info"]
                        if d["config_name"] != tag
                    ],
                    "configs": [
                        d for d in org_metadata["configs"] if d["config_name"] != tag
                    ],
                }
                metadata_update(
                    item.item_id,
                    new_metadata,
                    repo_type="dataset",
                    overwrite=True,
                )
            except EntryNotFoundError:
                pass


def delete_tagged_wandb_runs(tag):
    for run in WANDBAPI.runs(f"{WANDB_CONFIGS['team']}/{WANDB_CONFIGS['project']}"):
        if run.name.startswith(tag):
            run.delete()


def delete_all_huggingface_repos(collection_name):
    collection = get_collection(HUGGINGFACE_CONFIGS["collections"][collection_name])
    for item in collection.items:
        HFAPI.delete_repo(
            repo_id=item.item_id,
            repo_type="dataset" if collection_name != "models" else "model",
        )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.tag and script_args.clear_models:
        delete_tagged_huggingface_models(script_args.tag)
        print("Deleted all models with tag:", script_args.tag)

    if script_args.tag and script_args.clear_evaluations:
        delete_tagged_huggingface_evaluations(script_args.tag)
        print("Deleted all evaluations with tag:", script_args.tag)

    if script_args.tag and script_args.clear_wandb:
        delete_tagged_wandb_runs(script_args.tag)
        print("Deleted all wandb runs with tag:", script_args.tag)

    if script_args.clear_datasets:
        delete_all_huggingface_repos("datasets")
        print("Deleted all preprocessed datasets")

    if script_args.clear_all_hf:
        delete_all_huggingface_repos("datasets")
        delete_all_huggingface_repos("models")
        delete_all_huggingface_repos("evaluations")
        print("Deleted all huggingface repos")


if __name__ == "__main__":
    main()
