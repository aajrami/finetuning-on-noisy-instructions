# Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance


This repo contains the implementation code for the IJCNLP-AACL 2025 Main paper [Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance](https://arxiv.org/abs/2510.03528).

*Note: The implementation code is mainly based on the [open-instruct](https://github.com/allenai/open-instruct) code. Please also cite their work if you use this code.*

## Install Required packages
The required packages can be installed using the following command:
```
cd open-instruct/
pip install -r requirements.txt
```
## Dataset Preparation
### 1. Download and Format Datasets:
```
bash scripts/data/prepare_train_data.sh
```

### 2. Combine the instruction datasets in one JSON file
```
cd ..
python utils/combine_datasets.py
```

### 3. Perturb the instructions
```
python utils/perturb_instruction_dataset.py \
--input_file open-instruct/data/processed/SNI_Dolly_GPT4alpaca_processed_combined.jsonl \
--output_dir open-instruct/data/perturbed/
```

### 4. Create training epochs
We create different training sets with various percentage of perturbed\original instructions: 
```
python utils/create_training_dataset_mix.py \
--perturbed_dir open-instruct/data/processed/corrupted_dataset/ \
--original_dataset open-instruct/data/processed/SNI_Dolly_GPT4alpaca_processed_combined.jsonl --output_dir open-instruct/data/training/ \
--percentage_of_original <percentage>
```
The ```<percentage>``` can be: 0, 25, 50, 75 or 100.


## Fine-tuning small models using LoRA
```
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    /open-instruct/open_instruct/finetune.py \
    --model_name_or_path <model_name> \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name <model_name> \
    --use_slow_tokenizer False \
    --add_bos \
    --train_file <path_to_training_set_generated_from_step_4> \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir /path/to/output/dir/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 100
```

## Fine-tuning large models using QLoRA
```
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    /open-instruct/open_instruct/finetune.py \
    --model_name_or_path <model_name> \
    --gradient_checkpointing \
    --use_flash_attn \
    --use_qlora \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name <model_name> \
    --use_slow_tokenizer False \
    --add_bos \
    --train_file <path_to_training_set_generated_from_step_4> \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir /path/to/output/dir/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 100
```

## Adapter Merge
### Merge LoRA
```
python /open-instruct/open_instruct/merge_lora.py \
    --base_model_name_or_path <model_name> \
    --lora_model_name_or_path /path/to/saved/LoRA/dir/ \
    --output_dir /path/to/output/dir/ \
    --pad_to_multiple_of 8 \
    --save_tokenizer \
    --use_fast_tokenizer
```
### Merge QLoRA
```
python /open-instruct/open_instruct/merge_lora.py \
    --base_model_name_or_path <model_name> \
    --lora_model_name_or_path /path/to/saved/QLoRA/dir/ \
    --output_dir /path/to/output/dir/ \
    --pad_to_multiple_of 8 \
    --save_tokenizer \
    --use_fast_tokenizer \
    --qlora
```

## Evaluation
### MMLU
```
python /open-instruct/eval/mmlu/run_eval.py \
    --ntrain <0 or 5> \
    --data_dir /open-instruct/data/eval/mmlu \
    --save_dir /path/to/output/dir/ \
    --model_name_or_path <model_name_or_path> \
    --tokenizer_name_or_path <model_name_or_path> \
    --eval_batch_size 4 \
    --orig_examples_percentage <percentage> \
    --random_seed <seed>
```
The ```<percentage>``` can be 0, 0.25, 0.5, 0.75 and 1.0. Add ```--load_in_4bit``` when evaluating large models. 

### BBH
```
python /open-instruct/eval/bbh/run_eval.py \
    --data_dir /open-instruct/data/eval/bbh \
    --save_dir /path/to/output/dir/ \
    --model <model_name_or_path> \
    --tokenizer <model_name_or_path> \
    --use_vllm \
    --orig_examples_percentage <percentage> \
    --random_seed <seed>
```
The ```<percentage>``` can be 0, 0.25, 0.5, 0.75 and 1.0. Add ```--load_in_4bit``` when evaluating large models, and add ```no_cot``` when evaluating with direct (no chain-of-thought).

### GSM8K
```
python /open-instruct/eval/gsm/run_eval.py \
    --data_dir /open-instruct/data/eval/gsm/ \
    --save_dir /path/to/output/dir/ \
    --model <model_name_or_path> \
    --tokenizer <model_name_or_path> \
    --n_shot 8 \
    --use_vllm \
    --orig_examples_percentage <percentage> \
    --random_seed <seed>
```
The ```<percentage>``` can be 0, 0.25, 0.5, 0.75 and 1.0. Add ```--load_in_4bit``` when evaluating large models, and add ```no_cot``` when evaluating with direct (no chain-of-thought).

### ToxiGen
```
python /open-instruct/eval/toxigen/run_eval.py \
    --data_dir /open-instruct/data/eval/toxigen/ \
    --save_dir /path/to/output/dir/ \
    --model_name_or_path <model_name_or_path> \
    --use_vllm \
    --random_seed <seed>
```
Add ```--load_in_4bit``` when evaluating large models.

### TruthfulQA
```
python /open-instruct/eval/truthfulqa/run_eval.py \
    --data_dir /open-instruct/data/eval/truthfulqa/ \
    --save_dir /path/to/output/dir/ \
    --model_name_or_path <model_name_or_path> \
    --tokenizer_name_or_path <model_name_or_path> \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --random_seed <seed>
```
Add ```--load_in_4bit``` when evaluating large models.

## Citation  
```
@article{alajrami2025fine,
  title={Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance},
  author={Alajrami, Ahmed and Tan, Xingwei and Aletras, Nikolaos},
  journal={arXiv preprint arXiv:2510.03528},
  year={2025}
}
```