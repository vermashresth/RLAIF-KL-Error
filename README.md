# SAIL: Self-Improving Efficient Online Alignment of Large Language Models

## A lightweight plugin for Huggingface's DPOTrainer, achieving up to 11.6\% win-rate improvement with minimal overhead compared to fast DPO.


### Step 1: Filling YAML Configs
1. We need HuggingFace Hub and WanDB to manage experiments. Please fill in `./configs/services/hugggingface.yaml` and `./configs/services/wandb.yaml` with your acconut info.
2. We need OpenAI api to evaluate models. Please fill in `./configs/services/openai.yaml` with your account info.
3. We use a HuggingFace Space App to retrieve and review results. Please fill in `./viewer/.env` with your account info.


### Step 1.1
Install safe-rlhf repo

### Step 1.2
Login into huuggingface - !huggingface-cli login

### Step 2: Installation
1. With `python 3.10.*` and `CUDA 12.*` installed. You can run `python install -e .` to install this package called `cdpo`.

### Step 3: Config the Experiments to Run
1. Fill in or modify `./configs/tasks.yaml` for the set of experiments to run.

### Step 4: Run Experiments
1. Run command !cdpo prep then 
2. run `cdpo execute` to run all experiments specified.

### Step 5: View Results
1. Inside `./viewer` folder, run `streamlit run app.py` to start the result viewer. Using the UI there to analysis the results.

### Step 6: Evaluate Dataset with Bradley-Terry Probabilities
You can evaluate preference datasets and compute Bradley-Terry (BT) probabilities using the `evalreward_bt.py` script:

1. **Basic usage** - Evaluate a specific dataset split:
   ```bash
   python pipelines/evalreward_bt.py \
     --run_name your_experiment_name \
     --tag your_tag \
     --split default \
     --score_model_id PKU-Alignment/beaver-7b-v1.0-reward
   ```

2. **Process all splits** in the dataset:
   ```bash
   python pipelines/evalreward_bt.py \
     --run_name your_experiment_name \
     --tag your_tag \
     --process_all_splits \
     --score_model_id PKU-Alignment/beaver-7b-v1.0-reward
   ```

3. **Expected dataset format**: The dataset should have columns named `prompt`, `chosen`, and `rejected`.

4. **Output**: The script adds three columns to your dataset:
   - `chosen_score`: Reward score for the chosen response
   - `rejected_score`: Reward score for the rejected response  
   - `bt_prob`: Bradley-Terry probability computed as sigmoid(chosen_score - rejected_score)

5. **Troubleshooting**:
   - **ModuleNotFoundError: safe_rlhf**: Install the safe-rlhf library for PKU-Alignment models
   - **AttributeError: huggingface**: Configure `configs/services/huggingface.yaml` with your HuggingFace token
   - **Dataset not found**: Ensure your dataset is uploaded to HuggingFace Hub with the correct naming
   - **CUDA/Memory errors**: Reduce `--per_device_evalreward_batch_size` or use a smaller model

### Training time and memory requirements.
The approximate training time and memory requirements of each SAIL training on three models are: Qwen1.5-0.5B: 1-4 hours with 4*A40 GPUs; Phi-3-3.8B: 2-8 hours with 4*RTX6000Ada GPUs; Llama-3-8B: 2-12 hours with 4*A100 GPUs.

## Changes for debugigng training
Running on T4 1 GPU (in devices.yaml)
Batch size 1
load best model is off
Tasks.yaml has q0.5B model
use flash attention is False in dpo.py, sft.py, generate.py (pipelines)
