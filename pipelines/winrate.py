#!/usr/bin/env python3
"""
Winrate comp    reward_column_chosen: str = field(
        default="reward_score_chosen",
        metadata={"help": "Column name for chosen response rewards"}
    )
    reward_column_response: str = field(
        default="reward_score_generated", 
        metadata={"help": "Column name for generated response rewards"}
    )cript for evaluating different DPO runs.
Compares reward scores between chosen and generated responses across multiple runs.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset
from scipy.special import expit  # sigmoid function
import json
from transformers import HfArgumentParser

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
CACHE_CONFIGS = CONFIGS.utils.cache


@dataclass
class ScriptArguments:
    """Arguments for winrate comparison"""
    runs: List[str] = field(
        metadata={"help": "List of run names to compare"}
    )
    names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional display names for runs (must match length of runs)"}
    )
    tag: str = field(
        default="tag1",
        metadata={"help": "Dataset tag to evaluate"}
    )
    reward_column_chosen: str = field(
        default="reward_score",
        metadata={"help": "Column name for chosen response rewards"}
    )
    reward_column_response: str = field(
        default="reward_score_generated", 
        metadata={"help": "Column name for generated response rewards"}
    )
    cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "Cache directory for datasets"}
    )
    output_format: str = field(
        default="table",
        metadata={"help": "Output format: 'table', 'json', or 'csv'"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional output file to save results"}
    )
    sort_by: str = field(
        default="winrate",
        metadata={"help": "Sort results by: 'winrate', 'name', or 'none'"}
    )
    reverse_sort: bool = field(
        default=True,
        metadata={"help": "Sort in descending order (higher winrates first)"}
    )


def load_and_process_dataset(run_name: str, tag: str, cache_dir: str) -> pd.DataFrame:
    """Load dataset for a specific run and convert to pandas DataFrame"""
    try:
        dataset_name = f"{HUGGINGFACE_CONFIGS['prefix']['evaluations']}{run_name}"
        print(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, name=tag, cache_dir=cache_dir)
        
        # Use 'default' split if available, otherwise use first available split
        if "default" in dataset:
            split_data = dataset["default"]
        else:
            split_name = list(dataset.keys())[0]
            split_data = dataset[split_name]
            print(f"Warning: 'default' split not found, using '{split_name}'")
        
        df = split_data.to_pandas().sort_values("prompt").reset_index(drop=True)
        return df
    
    except Exception as e:
        print(f"Error loading dataset for run '{run_name}': {e}")
        return None


def calculate_winrate(df: pd.DataFrame, chosen_col: str, response_col: str) -> dict:
    """Calculate winrate and related statistics"""
    if chosen_col not in df.columns or response_col not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Required columns not found. Available columns: {available_cols}")
    
    r_chosen = df[chosen_col].values
    r_response = df[response_col].values
    
    # # Remove NaN values
    # mask = ~(np.isnan(r_chosen) | np.isnan(r_response))
    # r_chosen = r_chosen[mask]
    # r_response = r_response[mask]
    
    if len(r_chosen) == 0:
        return {
            "winrate": 0.0,
            "n_samples": 0,
            "avg_chosen_reward": 0.0,
            "avg_response_reward": 0.0,
            "avg_reward_diff": 0.0
        }
    
    # Calculate statistics
    winrate = np.mean(r_response > r_chosen) * 100
    n_samples = len(r_chosen)
    avg_chosen_reward = np.mean(r_chosen)
    avg_response_reward = np.mean(r_response)
    avg_reward_diff = avg_response_reward - avg_chosen_reward
    
    return {
        "winrate": winrate,
        "n_samples": n_samples,
        "avg_chosen_reward": avg_chosen_reward,
        "avg_response_reward": avg_response_reward,
        "avg_reward_diff": avg_reward_diff
    }


def compare_winrates(args: ScriptArguments) -> List[dict]:
    """Compare winrates across multiple runs"""
    results = []
    
    # Use provided names or default to run names
    display_names = args.names if args.names else args.runs
    
    if args.names and len(args.names) != len(args.runs):
        raise ValueError(f"Number of names ({len(args.names)}) must match number of runs ({len(args.runs)})")
    
    for run_name, display_name in zip(args.runs, display_names):
        print(f"\nProcessing run: {run_name}")
        
        # Load dataset
        df = load_and_process_dataset(run_name, args.tag, args.cache_dir)
        
        if df is None:
            print(f"Skipping run '{run_name}' due to loading error")
            continue
        
        # Calculate winrate
        stats = calculate_winrate(df, args.reward_column_chosen, args.reward_column_response)
        
        result = {
            "run_name": run_name,
            "display_name": display_name,
            **stats
        }
        results.append(result)
        
        print(f"  Winrate: {stats['winrate']:.2f}%")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Avg reward diff: {stats['avg_reward_diff']:.4f}")

    # Sort results if requested
    if args.sort_by == "winrate":
        results.sort(key=lambda x: x["winrate"], reverse=args.reverse_sort)
    elif args.sort_by == "name":
        results.sort(key=lambda x: x["display_name"], reverse=args.reverse_sort)
    
    return results


def format_output(results: List[dict], output_format: str) -> str:
    """Format results for output"""
    if not results:
        return "No results to display."
    
    if output_format == "json":
        return json.dumps(results, indent=2)
    
    elif output_format == "csv":
        df = pd.DataFrame(results)
        return df.to_csv(index=False)
    
    else:  # table format
        headers = [
            "Display Name",
            "Winrate (%)",
            "Samples",
            "Avg Chosen",
            "Avg Response", 
            "Reward Diff"
        ]
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        table_data = []
        for result in results:
            row = [
                result["display_name"],
                f"{result['winrate']:.2f}",
                str(result["n_samples"]),
                f"{result['avg_chosen_reward']:.4f}",
                f"{result['avg_response_reward']:.4f}",
                f"{result['avg_reward_diff']:.4f}"
            ]
            table_data.append(row)
            # Update column widths
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Create table
        lines = []
        
        # Header separator
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines.append(separator)
        
        # Header row
        header_row = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
        lines.append(header_row)
        lines.append(separator)
        
        # Data rows
        for row in table_data:
            data_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
            lines.append(data_row)
        
        lines.append(separator)
        
        return "\n".join(lines)


def main():
    print("=" * 60)
    print("Winrate Comparison Tool")
    print("=" * 60)
    
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    print(f"Comparing {len(script_args.runs)} runs:")
    for run in script_args.runs:
        print(f"  - {run}")
    print()
    
    # Compare winrates
    results = compare_winrates(script_args)
    
    if not results:
        print("No valid results found.")
        return
    
    # Format and display results
    output = format_output(results, script_args.output_format)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(output)
    
    # Save to file if requested
    if script_args.output_file:
        with open(script_args.output_file, 'w') as f:
            f.write(output)
        print(f"\nResults saved to: {script_args.output_file}")


if __name__ == "__main__":
    main()
