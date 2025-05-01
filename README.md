# SAIL: Self-Improving Efficient Online Alignment of Large Language Models

## A lightweight plugin for Huggingface's DPOTrainer, achieving up to 11.6\% win-rate improvement with minimal overhead compared to fast DPO.


### Step 1: Filling YAML Configs
1. We need HuggingFace Hub and WanDB to manage experiments. Please fill in `./configs/services/hugggingface.yaml` and `./configs/services/wandb.yaml` with your acconut info.
2. We need OpenAI api to evaluate models. Please fill in `./configs/services/openai.yaml` with your account info.
3. We use a HuggingFace Space App to retrieve and review results. Please fill in `./viewer/.env` with your account info.

### Step 2: Installation
1. With `python 3.10.*` and `CUDA 12.*` installed. You can run `python install -e .` to install this package called `cdpo`.

### Step 3: Config the Experiments to Run
1. Fill in or modify `./configs/tasks.yaml` for the set of experiments to run.

### Step 4: Run Experiments
1. Run command `cdpo execute` to run all experiments specified.

### Step 5: View Results
1. Inside `./viewer` folder, run `streamlit run app.py` to start the result viewer. Using the UI there to analysis the results.


### Training time and memory requirements.
The approximate training time and memory requirements of each SAIL training on three models are: Qwen1.5-0.5B: 1-4 hours with 4*A40 GPUs; Phi-3-3.8B: 2-8 hours with 4*RTX6000Ada GPUs; Llama-3-8B: 2-12 hours with 4*A100 GPUs.