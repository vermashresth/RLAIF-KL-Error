import os
import requests
from io import StringIO
import re
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, get_collection
from huggingface_hub.utils import RevisionNotFoundError
import wandb
from datasets import load_dataset, get_dataset_config_names
from joblib import Memory
import plotly.express as px
import streamlit as st

load_dotenv()
HFAPI = HfApi(token=os.environ["HF_TOKEN"])
WANDBAPI = wandb.Api(api_key=os.environ["WANDB_TOKEN"])
MEMORY = Memory(location=os.environ["JOBLIB_CACHE_DIR"], verbose=0)
TAGS = []
DISPLAY_NAMES = {
    "val_loss": "Val Loss",
}


def short_scientific(x):
    if x == 0:
        return "0e0"
    exponent = int(f"{x:.0e}".split("e")[1])
    mantissa = x / 10**exponent
    # First convert the formatted float string to a float, then to an int
    mantissa = int(float(f"{mantissa:.1f}"))
    return f"{mantissa}e{exponent}"


def parse_parameters(param_str):
    # Dictionary to map greek letter names to unicode
    greek_letters = {
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
        "zeta": "ζ",
        "eta": "η",
        "theta": "θ",
        "iota": "ι",
        "kappa": "κ",
        "lambda": "λ",
        "mu": "μ",
        "nu": "ν",
        "xi": "ξ",
        "omicron": "ο",
        "pi": "π",
        "rho": "ρ",
        "sigma": "σ",
        "tau": "τ",
        "upsilon": "υ",
        "phi": "φ",
        "chi": "χ",
        "psi": "ψ",
        "omega": "ω",
    }
    # Regular expression to match key-value pairs
    pattern = re.compile(r"([a-zA-Z]+)(\d+\.?\d*)")
    matches = pattern.findall(param_str)
    result = []
    for key, value in matches:
        # Replace Greek letters
        key = greek_letters.get(key, key)
        # Convert numbers to float, then format them
        if value == "0":
            value = "no"
        elif value == "1":
            value = "yes"
        else:
            num = float(value)
            if num.is_integer():
                num = int(num)
                if num >= 100 or num <= -100:
                    value = short_scientific(num)
                else:
                    value = str(num)
            else:
                if num < 0.01 and num > -0.01:
                    value = short_scientific(num)
                else:
                    value = f"{num:.2g}"
        result.append(f"{key}={value}")
    return result


def get_run_duration(run_name, tag):
    runs = WANDBAPI.runs(os.environ["WANDB_PROJECT"])
    for run in runs:
        if run.name == tag + "-" + run_name:
            duration = run.summary.get("_runtime", np.nan)
            return int(duration)
    return np.nan


@MEMORY.cache
def get_eval_tag(eval_id, last_modified):
    tags = get_dataset_config_names(eval_id)
    return tags


@MEMORY.cache
def get_model_tag(model_id, last_modified):
    refs = HFAPI.list_repo_refs(model_id)
    return [branch.name for branch in refs.branches]


def get_all_tags(pbar):
    tags = []

    # Add tags from evalutaions
    eval_collection = get_collection(os.environ["EVALUTAION_COLLECTION_SLUG"])
    counter = 0
    total = len(eval_collection.items)
    for item in eval_collection.items:
        eval_id = item.item_id
        eval_last_modified = HFAPI.repo_info(
            eval_id, repo_type="dataset"
        ).lastModified.isoformat()
        tags.extend(get_eval_tag(eval_id, eval_last_modified))
        counter += 1
        pbar.progress(counter / total * 0.5, text="Retrieving tags")

    # Add tags from models
    model_collection = get_collection(os.environ["MODEL_COLLECTION_SLUG"])
    counter = 0
    total = len(model_collection.items)
    for item in model_collection.items:
        model_id = item.item_id
        model_last_modified = HFAPI.repo_info(
            model_id, repo_type="model"
        ).lastModified.isoformat()
        tags.extend(get_model_tag(model_id, model_last_modified))
        counter += 1
        pbar.progress(counter / total * 0.5 + 0.5, text="Retrieving tags")

    # Sanitize tags
    tags = sorted(list(set(tags) - set(["main"])))
    return tags


@MEMORY.cache
def get_eval_scores(eval_id, last_modified):
    tags = get_dataset_config_names(eval_id)
    eval_scores = {}
    for tag in tags:
        eval_dataset = load_dataset(
            eval_id, tag, split="default", cache_dir=os.environ["DATASET_CACHE_DIR"]
        )
        eval_dataframe = eval_dataset.to_pandas()
        reward = (
            eval_dataframe[~eval_dataframe["reward_score"].isna()][
                "reward_score"
            ].mean()
            if "reward_score" in eval_dataframe.columns
            else np.nan
        )
        winrate = (
            (
                eval_dataframe[~eval_dataframe["gpt_score"].isna()]["gpt_score"] > 0
            ).mean()
            if "gpt_score" in eval_dataframe.columns
            else np.nan
        )
        eval_scores[tag] = (reward, winrate)
    return eval_scores


@MEMORY.cache
def get_model_scores(model_id, tag, last_modified):
    # Load the model card from a specified repository
    text = requests.get(f"https://huggingface.co/{model_id}/raw/{tag}/README.md").text

    # Check if the model card is empty
    if text.startswith("Invalid rev id:"):
        return (np.nan, np.nan, np.nan)

    # Use pandas to directly read the Markdown table
    # We find the table by splitting the text and isolating the portion containing the Markdown table
    try:
        # Find the start of the table by searching for the header row
        start = text.index("| Training Loss |")  # Locate the start of the table
        end = (
            text.rfind("|", start) + 1
        )  # Find the last pipe character after the start of the table

        markdown_table = text[start:end]

        # Use StringIO to simulate a file-like object, which pandas can read from
        data = pd.read_csv(StringIO(markdown_table), sep="|", engine="python")
        data = data.dropna(
            axis=1, how="all"
        )  # Drop columns that are all NaN due to markdown pipe characters on edges
        data = data.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x
        )  # Strip whitespace from all string entries
        data = data.drop(0)  # Drop the seperator row
        data = data.astype(float)  # Convert all columns to float

        return (
            data[" Validation Loss "].astype(float).tolist(),
            data[" Rewards/accuracies "].astype(float).tolist(),
            data[" Rewards/margins "].astype(float).tolist(),
        )
    except ValueError:
        return ([np.nan], [np.nan], [np.nan])


def prepare_result_dataframe(tags, pbar):
    eval_collection = get_collection(os.environ["EVALUTAION_COLLECTION_SLUG"])
    dd = []
    counter = 0
    total = (1 + len(tags)) * len(eval_collection.items)

    for item in eval_collection.items:
        # Get the eval and model id
        eval_id = item.item_id
        run_name = eval_id[len(os.environ["EVALUTAION_COLLECTION_SLUG"]) :]
        model_id = os.environ["MODEL_COLLECTION_SLUG"] + run_name

        # Eval scores are fetched for all tags at once
        eval_last_modified = HFAPI.repo_info(
            eval_id, repo_type="dataset"
        ).lastModified.isoformat()
        eval_scores = get_eval_scores(eval_id, eval_last_modified)
        counter += 1
        pbar.progress(counter / total, text="Fetching results")

        for tag in tags:
            # Model scores are fetched per tag
            try:
                model_last_modified = HFAPI.repo_info(
                    model_id, revision=tag, repo_type="model"
                ).lastModified.isoformat()
                model_scores = get_model_scores(model_id, tag, model_last_modified)
            except RevisionNotFoundError:
                model_scores = ([np.nan], [np.nan], [np.nan])
            counter += 1
            pbar.progress(counter / total, text="Fetching results")

            # Add to table if the tag is in the eval scores
            if tag in eval_scores:
                dd.append(
                    {
                        # Indices
                        "run_name": run_name,
                        "tag": tag,
                        # Main table
                        "setup_name": run_name.split("_")[0].split("-")[-1],
                        "para_list": parse_parameters(run_name.split("_")[-1]),
                        "model_link": f"https://huggingface.co/{os.environ['MODEL_COLLECTION_SLUG']}{run_name}/blob/{tag}/README.md",
                        "dataset_link": f"https://huggingface.co/datasets/{os.environ['DATASET_COLLECTION_SLUG']}{run_name.split('_')[2]}",
                        "evaluation_link": f"https://huggingface.co/datasets/{os.environ['EVALUTAION_COLLECTION_SLUG']}{run_name}/viewer/{tag}",
                        "duration": get_run_duration(run_name, tag),
                        "val_loss": model_scores[0][-1],
                        "reward_acc": model_scores[1][-1],
                        "reward_margin": model_scores[2][-1],
                        "ref_reward": eval_scores[tag][0],
                        "winrate": eval_scores[tag][1] * 100,
                        # Learning curve table
                        "val_loss_history": model_scores[0],
                        "reward_acc_history": model_scores[1],
                        "reward_margin_history": model_scores[2],
                        # Modification info table
                        "eval_last_modified": datetime.fromisoformat(
                            eval_last_modified
                        ),
                        "model_last_modified": datetime.fromisoformat(
                            model_last_modified
                        ),
                    }
                )
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(
        dd,
        columns=[
            # Indices
            "run_name",
            "tag",
            # Main table
            "setup_name",
            "para_list",
            "model_link",
            "dataset_link",
            "evaluation_link",
            "duration",
            "val_loss",
            "reward_acc",
            "reward_margin",
            "ref_reward",
            "winrate",
            # Learning curve table
            "val_loss_history",
            "reward_acc_history",
            "reward_margin_history",
            # Modification info table
            "eval_last_modified",
            "model_last_modified",
        ],
    )
    return df


def main():
    global TAGS

    st.set_page_config(layout="wide")  # Set the layout to "wide" mode

    # Sidebar
    with st.sidebar:
        # General progress bar
        pbar = st.progress(0, text="Progress")
        st.divider()

        # Clear the cache
        if st.button("Clear all cache"):
            MEMORY.clear(warn=False)
            TAGS = []

        # Get all tags
        if st.button("Get all experiment tags") or (not TAGS):
            TAGS = get_all_tags(pbar)
        st.divider()

        is_aggregate = st.checkbox("Aggregate experiments", value=False)
        aggregate_method = st.selectbox("Aggregate method:", ["average", "best"])
        st.divider()

        tags = st.multiselect("Experiment tags:", TAGS, [])

    # Load the data based on the selected tag
    df = prepare_result_dataframe(tags, pbar)

    # Main table
    st.title("Experiment Results")
    # Hide columns
    df_main = df.copy()
    df_main = df_main[
        [
            "run_name",
            "setup_name",
            "para_list",
            "model_link",
            "dataset_link",
            "evaluation_link",
            "duration",
            "val_loss",
            "reward_acc",
            "reward_margin",
            "ref_reward",
            "winrate",
        ]
    ]
    # Aggregate the main table
    if is_aggregate:
        aggregate_method = "mean" if aggregate_method == "average" else "max"
        df_main = (
            df_main.groupby(["run_name"])
            .agg(
                {
                    "setup_name": "first",
                    "para_list": "first",
                    "model_link": lambda x: x.iloc[
                        x.index.get_loc(df["val_loss"].idxmin())
                    ],
                    "dataset_link": lambda x: x.iloc[
                        x.index.get_loc(df["val_loss"].idxmin())
                    ],
                    "evaluation_link": lambda x: x.iloc[
                        x.index.get_loc(df["val_loss"].idxmin())
                    ],
                    "duration": aggregate_method,
                    "val_loss": "min" if aggregate_method == "max" else "mean",
                    "reward_acc": aggregate_method,
                    "reward_margin": aggregate_method,
                    "ref_reward": aggregate_method,
                    "winrate": aggregate_method,
                }
            )
            .reset_index()
        )
    # Sort the main table and drop the run_name column
    df_main = df_main.sort_values(by=["run_name"]).drop(columns=["run_name"])
    # Adjust the dataframe display
    st.dataframe(
        df_main,
        column_config={
            "setup_name": st.column_config.TextColumn(
                "Setup",
                help="Setup of the run.",
                disabled=True,
            ),
            "para_list": st.column_config.ListColumn(
                "Parameters",
                help="Parameters of the run.",
            ),
            "model_link": st.column_config.LinkColumn(
                "Model",
                help="Link to the model repo on Huggingface.",
                disabled=True,
                display_text="^[^_]*_([^_]*)_",
            ),
            "dataset_link": st.column_config.LinkColumn(
                "Data",
                help="Link to the dataset repo on Huggingface.",
                disabled=True,
                display_text="(?:[^-]*-){3}(.*)$",
            ),
            "evaluation_link": st.column_config.LinkColumn(
                "Resp",
                help="Link to the generated responses and evaluations repo on Huggingface.",
                disabled=True,
                display_text="link",
            ),
            "duration": st.column_config.NumberColumn(
                "Duration",
                help="Duration of run in seconds.",
                disabled=True,
                format="%d",
            ),
            "val_loss": st.column_config.NumberColumn(
                "Val Loss",
                help="Validation loss of the model.",
                disabled=True,
                format="%.4f",
            ),
            "reward_acc": st.column_config.NumberColumn(
                "Rwd Acc",
                help="Validation reward accuracy of the model.",
                disabled=True,
                format="%.4f",
            ),
            "reward_margin": st.column_config.NumberColumn(
                "Rwd Marg",
                help="Validation reward margin of the model.",
                disabled=True,
                format="%.4f",
            ),
            "ref_reward": st.column_config.ProgressColumn(
                "Reference Reward",
                # width="medium",
                help="Average reward of response evaluated by reference reward model.",
                min_value=df_main["ref_reward"].min() - 3,
                max_value=df_main["ref_reward"].max() + 3,
                format="%.2f",
            ),
            "winrate": st.column_config.ProgressColumn(
                "Winrate",
                # width="medium",
                help="Winrate of response vs. chosen answer evaluated by GPT.",
                min_value=0,
                max_value=100,
                format="%.2f%%",
            ),
        },
        width=2000,  # Adjust the width as needed
        hide_index=True,
    )

    df_csv = df.drop(
        columns=[
            "setup_name",
            "para_list",
            "model_link",
            "dataset_link",
            "evaluation_link",
            "eval_last_modified",
            "model_last_modified",
        ]
    ).to_csv(index=False)
    st.download_button(
        label="Download Table CSV",
        data=df_csv,
        file_name="table.csv",
        mime="text/csv",
    )

    st.divider()

    st.header("Other Information")
    tab_heetmap, tab_curves, tab_modify = st.tabs(
        ["Parameter Tuning", "Learning Curves", "Modification Info"]
    )

    with tab_heetmap:
        df_heatmap = df.copy()
        heatmap_col1, heatmap_col2 = st.columns(2)
        with heatmap_col1:
            heatmap_setup_name = st.selectbox(
                "Select setup name",
                [sn for sn in df_heatmap.setup_name.unique() if sn != "DPO"],
            )
        with heatmap_col2:
            heatmap_metric = st.selectbox(
                "Select metric",
                ["Duration", "Reward Margin", "Reference Reward", "Winrate"],
            )
            heatmap_metric = {
                "Duration": "duration",
                "Reward Margin": "reward_margin",
                "Reference Reward": "ref_reward",
                "Winrate": "winrate",
            }[heatmap_metric]
        param_map = {"DDP": ("r", "ρ"), "DPP": ("p", "π"), "DPR": ("g", "γ")}
        x_param, y_param = param_map[heatmap_setup_name]
        df_filtered = df_heatmap[df_main["setup_name"] == heatmap_setup_name].copy()
        df_filtered[x_param] = df_filtered["para_list"].apply(
            lambda x: float(
                [p.split("=")[1] for p in x if p.split("=")[0] == x_param][0]
            )
        )
        df_filtered[y_param] = df_filtered["para_list"].apply(
            lambda x: float(
                [p.split("=")[1] for p in x if p.split("=")[0] == y_param][0]
            )
        )
        if heatmap_metric == "duration":
            df_filtered["duration"] = df_filtered["duration"] * (
                1 - df_filtered[x_param]
            )
        heatmap_data = df_filtered.pivot_table(
            index=y_param, columns=x_param, values=heatmap_metric
        )
        fig = px.imshow(
            heatmap_data,
            labels=dict(x=x_param, y=y_param, color=heatmap_metric),
            color_continuous_scale="Viridis",
            aspect="equal",
        )
        fig.update_traces(
            zmin=heatmap_data.min().min(),
            zmax=heatmap_data.max().max(),
            colorscale="Viridis",
        )
        fig.update_layout(
            coloraxis_colorbar_title_text={
                "duration": "Duration",
                "reward_margin": "Reward Margin",
                "ref_reward": "Reference Reward",
                "winrate": "Winrate",
            }[heatmap_metric]
        )
        fig.update_xaxes(
            tickvals=df_filtered[x_param].unique(),
            ticktext=df_filtered[x_param].unique().astype(str),
            title=x_param,
        )
        fig.update_yaxes(
            tickvals=df_filtered[y_param].unique(),
            ticktext=df_filtered[y_param].unique().astype(str),
            title=y_param,
            range=[max(df_filtered[y_param]), min(df_filtered[y_param])],
        )
        st.plotly_chart(fig)

        fig_json = fig.to_json()
        st.download_button(
            label="Download Figure JSON",
            data=fig_json,
            file_name="figure.json",
            mime="application/json",
        )

    with tab_curves:
        df_curves = df.copy()
        df_curves = df_curves[
            [
                "run_name",
                "tag",
                "val_loss_history",
                "reward_acc_history",
                "reward_margin_history",
            ]
        ]
        st.dataframe(
            df_curves,
            column_config={
                "run_name": st.column_config.TextColumn(
                    "Run Name",
                    help="Name of the run.",
                    disabled=True,
                ),
                "tag": st.column_config.TextColumn(
                    "Tag",
                    help="Tag of the run.",
                    disabled=True,
                ),
                "val_loss_history": st.column_config.LineChartColumn(
                    "Validation Loss",
                    width="medium",
                    help="Validation loss curve of the model.",
                    y_min=(
                        df_curves["val_loss_history"].explode().min()
                        if not df_curves.empty
                        else 0
                    ),
                    y_max=(
                        df_curves["val_loss_history"].explode().max()
                        if not df_curves.empty
                        else 1
                    ),
                ),
                "reward_acc_history": st.column_config.LineChartColumn(
                    "Reward Accuracy",
                    width="medium",
                    help="Validation reward accuracy curve of the model.",
                    y_min=(
                        df_curves["reward_acc_history"].explode().min()
                        if not df_curves.empty
                        else 0
                    ),
                    y_max=(
                        df_curves["reward_acc_history"].explode().max()
                        if not df_curves.empty
                        else 1
                    ),
                ),
                "reward_margin_history": st.column_config.LineChartColumn(
                    "Reward Margin",
                    width="medium",
                    help="Validation reward margin curve of the model.",
                    y_min=(
                        df_curves["reward_margin_history"].explode().min()
                        if not df_curves.empty
                        else 0
                    ),
                    y_max=(
                        df_curves["reward_margin_history"].explode().max()
                        if not df_curves.empty
                        else 1
                    ),
                ),
            },
            width=2000,  # Adjust the width as needed
            hide_index=True,
        )

    with tab_modify:
        df_modify = df.copy()
        df_modify = df_modify[
            [
                "run_name",
                "tag",
                "eval_last_modified",
                "model_last_modified",
            ]
        ]
        st.dataframe(
            df_modify,
            column_config={
                "run_name": st.column_config.TextColumn(
                    "Run Name",
                    help="Name of the run.",
                    disabled=True,
                ),
                "tag": st.column_config.TextColumn(
                    "Tag",
                    help="Tag of the run.",
                    disabled=True,
                ),
                "eval_last_modified": st.column_config.DatetimeColumn(
                    "Eval Last Modified",
                    help="Last modified date of the evaluation.",
                    disabled=True,
                    format="MMM D, h:mm a",
                    timezone="America/New_York",
                ),
                "model_last_modified": st.column_config.DatetimeColumn(
                    "Model Last Modified",
                    help="Last modified date of the model.",
                    disabled=True,
                    format="MMM D, h:mm a",
                    timezone="America/New_York",
                ),
            },
            width=2000,  # Adjust the width as needed
            hide_index=True,
        )


if __name__ == "__main__":
    main()
