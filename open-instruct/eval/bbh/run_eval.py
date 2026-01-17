import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import vllm
import evaluate

from transformers import BertTokenizer, BertForMaskedLM


import sys
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from utils import (
    load_hf_lm,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata
)

from utils import (
    shuffle_words, 
    remove_stop_words,
    replace_words_using_bert,
    insert_words_using_bert,
    introduce_typos,
    delete_words,
)

exact_match = evaluate.load("exact_match")

perturbation_func = {
    "shuffle_words": shuffle_words,
    "remove_stop_words": remove_stop_words,
    "replace_words_using_bert": replace_words_using_bert,
    "insert_words_using_bert": insert_words_using_bert,
    "introduce_typos": introduce_typos,
    "delete_words": delete_words,
}

# Load pre-trained BERT model and tokenizer for perturbation functions
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_model.eval()

def main(args):
    random.seed(args.random_seed)

    if args.orig_examples_percentage < 1.0:
        print(f"Perturbing the instruction text using mix perturbation strategy. {args.orig_examples_percentage * 100}% of examples kept as original with no perturbation. {(1 - args.orig_examples_percentage) * 100}% of examples splitted equally between the 6 different perurbation strategies.")    

    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    if args.orig_examples_percentage < 1.0:
        all_prompts_sws = {}
        all_prompts_rsws = {}
        all_prompts_repws = {}
        all_prompts_insws = {}
        all_prompts_intypos = {}
        all_prompts_dws = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.orig_examples_percentage < 1.0:
                task_prompt_sws = task_prompt
                task_prompt_rsws = task_prompt
                task_prompt_repws = task_prompt
                task_prompt_insws = task_prompt
                task_prompt_intypos = task_prompt
                task_prompt_dws = task_prompt
            ### perturb prompt question
            prompt_fields = task_prompt.split("\n\n")
            if args.orig_examples_percentage < 1.0:
                prompt_fields_sws = prompt_fields.copy()
                prompt_fields_rsws = prompt_fields.copy()
                prompt_fields_repws = prompt_fields.copy()
                prompt_fields_insws = prompt_fields.copy()
                prompt_fields_intypos = prompt_fields.copy()
                prompt_fields_dws = prompt_fields.copy()
                prompt_fields_sws[0] = perturbation_func["shuffle_words"](prompt_fields[0], random_seed=args.random_seed)
                prompt_fields_rsws[0] = perturbation_func["remove_stop_words"](prompt_fields[0], random_seed=args.random_seed)
                prompt_fields_repws[0] = perturbation_func["replace_words_using_bert"](prompt_fields[0], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                prompt_fields_insws[0] = perturbation_func["insert_words_using_bert"](prompt_fields[0], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                prompt_fields_intypos[0] = perturbation_func["introduce_typos"](prompt_fields[0], random_seed=args.random_seed)
                prompt_fields_dws[0] = perturbation_func["delete_words"](prompt_fields[0], random_seed=args.random_seed)
                new_prompt_fields_sws = []
                new_prompt_fields_rsws = []
                new_prompt_fields_repws = []
                new_prompt_fields_insws = []
                new_prompt_fields_intypos = []
                new_prompt_fields_dws = []
            for prompt_field in prompt_fields:
                if prompt_field.startswith("Q:"):
                    prompt_parts = prompt_field.split("\n")
                    if args.orig_examples_percentage < 1.0:
                        prompt_parts_sws = prompt_parts.copy()
                        prompt_parts_rsws = prompt_parts.copy()
                        prompt_parts_repws = prompt_parts.copy()
                        prompt_parts_insws = prompt_parts.copy()
                        prompt_parts_intypos = prompt_parts.copy()
                        prompt_parts_dws = prompt_parts.copy()
                        prompt_question_sws = perturbation_func["shuffle_words"](prompt_parts[0][3:], random_seed=args.random_seed)
                        prompt_question_rsws = perturbation_func["remove_stop_words"](prompt_parts[0][3:], random_seed=args.random_seed)
                        prompt_question_repws = perturbation_func["replace_words_using_bert"](prompt_parts[0][3:], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                        prompt_question_insws = perturbation_func["insert_words_using_bert"](prompt_parts[0][3:], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                        prompt_question_intypos = perturbation_func["introduce_typos"](prompt_parts[0][3:], random_seed=args.random_seed)
                        prompt_question_dws = perturbation_func["delete_words"](prompt_parts[0][3:], random_seed=args.random_seed)
                        prompt_parts_sws[0] = "Q: " + prompt_question_sws
                        prompt_parts_rsws[0] = "Q: " + prompt_question_rsws
                        prompt_parts_repws[0] = "Q: " + prompt_question_repws
                        prompt_parts_insws[0] = "Q: " + prompt_question_insws
                        prompt_parts_intypos[0] = "Q: " + prompt_question_intypos
                        prompt_parts_dws[0] = "Q: " + prompt_question_dws
                        if not prompt_parts[1].startswith("Options:") and not prompt_parts[1].startswith("A:"):
                            prompt_parts_sws[1] = perturbation_func["shuffle_words"](prompt_parts[1], random_seed=args.random_seed)
                            prompt_parts_rsws[1] = perturbation_func["remove_stop_words"](prompt_parts[1], random_seed=args.random_seed)
                            prompt_parts_repws[1] = perturbation_func["replace_words_using_bert"](prompt_parts[1], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                            prompt_parts_insws[1] = perturbation_func["insert_words_using_bert"](prompt_parts[1], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                            prompt_parts_intypos[1] = perturbation_func["introduce_typos"](prompt_parts[1], random_seed=args.random_seed)
                            prompt_parts_dws[1] = perturbation_func["delete_words"](prompt_parts[1], random_seed=args.random_seed)
                        prompt_field_sws = "\n".join(prompt_parts_sws)
                        prompt_field_rsws = "\n".join(prompt_parts_rsws)
                        prompt_field_repws = "\n".join(prompt_parts_repws)
                        prompt_field_insws = "\n".join(prompt_parts_insws)
                        prompt_field_intypos = "\n".join(prompt_parts_intypos)
                        prompt_field_dws = "\n".join(prompt_parts_dws)
                        new_prompt_fields_sws.append(prompt_field_sws)
                        new_prompt_fields_rsws.append(prompt_field_rsws)
                        new_prompt_fields_repws.append(prompt_field_repws)
                        new_prompt_fields_insws.append(prompt_field_insws)
                        new_prompt_fields_intypos.append(prompt_field_intypos)
                        new_prompt_fields_dws.append(prompt_field_dws)
            if args.orig_examples_percentage < 1.0:
                task_prompt_sws = "\n\n".join(new_prompt_fields_sws)
                task_prompt_rsws = "\n\n".join(new_prompt_fields_rsws)
                task_prompt_repws = "\n\n".join(new_prompt_fields_repws)
                task_prompt_insws = "\n\n".join(new_prompt_fields_insws)
                task_prompt_intypos = "\n\n".join(new_prompt_fields_intypos)
                task_prompt_dws = "\n\n".join(new_prompt_fields_dws)

            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
                if args.orig_examples_percentage < 1.0:
                    # shuffled words
                    prompt_fields_sws = task_prompt_sws.split("\n\n")
                    new_prompt_fields_sws = []
                    for prompt_field in prompt_fields_sws:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_sws.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_sws.append(prompt_field)
                    task_prompt_sws = "\n\n".join(new_prompt_fields_sws)
                    # remove stop words
                    prompt_fields_rsws = task_prompt_rsws.split("\n\n")
                    new_prompt_fields_rsws = []
                    for prompt_field in prompt_fields_rsws:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_rsws.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_rsws.append(prompt_field)
                    task_prompt_rsws = "\n\n".join(new_prompt_fields_rsws)

                    # replace_words_using_bert
                    prompt_fields_repws = task_prompt_repws.split("\n\n")
                    new_prompt_fields_repws = []
                    for prompt_field in prompt_fields_repws:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_repws.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_repws.append(prompt_field)
                    task_prompt_repws = "\n\n".join(new_prompt_fields_repws)

                    # insert_words_using_bert
                    prompt_fields_insws = task_prompt_insws.split("\n\n")
                    new_prompt_fields_insws = []
                    for prompt_field in prompt_fields_insws:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_insws.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_insws.append(prompt_field)
                    task_prompt_insws = "\n\n".join(new_prompt_fields_insws)

                    # introduce_typos
                    prompt_fields_intypos = task_prompt_intypos.split("\n\n")
                    new_prompt_fields_intypos = []
                    for prompt_field in prompt_fields_intypos:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_intypos.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_intypos.append(prompt_field)
                    task_prompt_intypos = "\n\n".join(new_prompt_fields_intypos)

                    # delete_words
                    prompt_fields_dws = task_prompt_dws.split("\n\n")
                    new_prompt_fields_dws = []
                    for prompt_field in prompt_fields_dws:
                        if prompt_field.startswith("Q:"):
                            assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                            assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                            answer = prompt_field.split("So the answer is")[-1].strip()
                            question = prompt_field.split("\nA:")[0].strip()
                            new_prompt_fields_dws.append(question + "\nA: " + answer)
                        else:
                            new_prompt_fields_dws.append(prompt_field)
                    task_prompt_dws = "\n\n".join(new_prompt_fields_dws)

            all_prompts[task_name] = task_prompt
            if args.orig_examples_percentage < 1.0:
                all_prompts_sws[task_name] = task_prompt_sws
                all_prompts_rsws[task_name] = task_prompt_rsws
                all_prompts_repws[task_name] = task_prompt_repws
                all_prompts_insws[task_name] = task_prompt_insws
                all_prompts_intypos[task_name] = task_prompt_intypos
                all_prompts_dws[task_name] = task_prompt_dws

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    # Load model if not using OpenAI API
    if args.model_name_or_path:
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if args.use_vllm:
            print("Loading vllm model...")
            if args.load_in_4bit:
                model = vllm.LLM(
                    model=args.model_name_or_path,
                    quantization="bitsandbytes",
                    load_format= "bitsandbytes",
                    # max_model_len=10240,
                    tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                    tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                    tensor_parallel_size=torch.cuda.device_count(),
                    tokenizer_revision=args.hf_revision,
                    revision=args.hf_revision,
                )
            else:
                model = vllm.LLM(
                    model=args.model_name_or_path,
                    tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                    tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                    tensor_parallel_size=torch.cuda.device_count(),
                    tokenizer_revision=args.hf_revision,
                    revision=args.hf_revision,
                )
        else:
            print("Loading model and tokenizer with huggingface...")
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path, 
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]
        if args.orig_examples_percentage < 1.0:
            task_prompt_sws = all_prompts_sws[task_name]
            task_prompt_rsws = all_prompts_rsws[task_name]
            task_prompt_repws = all_prompts_repws[task_name]
            task_prompt_insws = all_prompts_insws[task_name]
            task_prompt_intypos = all_prompts_intypos[task_name]
            task_prompt_dws = all_prompts_dws[task_name]


            n_examples = len(task_examples)
        
            # Calculate number of examples for each perturbation
            n_perturbation = int(n_examples * (1 - args.orig_examples_percentage)/6) # we have 6 perturbation methods
        
            
            # Randomly select indices for modifications
            all_indices = list(range(n_examples))
            random.shuffle(all_indices)
            
            # Get indices for each modification
            shuffle_indices = all_indices[:n_perturbation]
            remove_stop_indices = all_indices[n_perturbation:n_perturbation * 2]
            replace_words_indices = all_indices[n_perturbation * 2:n_perturbation * 3]
            insert_words_indices = all_indices[n_perturbation * 3:n_perturbation * 4]
            introduce_typos_indices = all_indices[n_perturbation * 4:n_perturbation * 5]
            delete_words_indices = all_indices[n_perturbation * 5:n_perturbation * 6]
                    
            print("\n")
            print("***** shuffle_indices *****",shuffle_indices)
            print("***** remove_stop_indices *****",remove_stop_indices)
            print("***** replace_words_indices *****",replace_words_indices)
            print("***** insert_words_indices *****",insert_words_indices)
            print("***** introduce_typos_indices *****",introduce_typos_indices)
            print("***** delete_words_indices *****",delete_words_indices)

            # Apply shuffling
            for idx in shuffle_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["shuffle_words"](example_input_parts[0], random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)
                
            # Apply stop words removal
            for idx in remove_stop_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["remove_stop_words"](example_input_parts[0], random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)

            # Apply replace word
            for idx in replace_words_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["replace_words_using_bert"](example_input_parts[0], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)

            # Apply insert words
            for idx in insert_words_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["insert_words_using_bert"](example_input_parts[0], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)

            # Apply introduce typos
            for idx in introduce_typos_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["introduce_typos"](example_input_parts[0], random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)

            # Apply delete words
            for idx in delete_words_indices:
                example_input_parts = task_examples[idx]["input"].split("\n")
                example_input_parts[0] = perturbation_func["delete_words"](example_input_parts[0], random_seed=args.random_seed)
                task_examples[idx]["input"] = "\n".join(example_input_parts)

        if args.model_name_or_path:
            # prepare prompts    
            if args.use_chat_format:
                prompts = []
                chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
                for i in range(len(task_examples)):
                    if args.orig_examples_percentage < 1.0:
                        if i in shuffle_indices:
                            prompt = task_prompt_sws.strip() + "\n\nQ: " + task_examples[i]["input"]
                        elif i in remove_stop_indices:
                            prompt = task_prompt_rsws.strip() + "\n\nQ: " + task_examples[i]["input"]
                        elif i in replace_words_indices:
                            prompt = task_prompt_repws.strip() + "\n\nQ: " + task_examples[i]["input"]
                        elif i in insert_words_indices:
                            prompt = task_prompt_insws.strip() + "\n\nQ: " + task_examples[i]["input"]
                        elif i in introduce_typos_indices:
                            prompt = task_prompt_intypos.strip() + "\n\nQ: " + task_examples[i]["input"]
                        elif i in delete_words_indices:
                            prompt = task_prompt_dws.strip() + "\n\nQ: " + task_examples[i]["input"]
                        else:
                            prompt = task_prompt.strip() + "\n\nQ: " + task_examples[i]["input"]
                    else:
                        prompt = task_prompt.strip() + "\n\nQ: " + task_examples[i]["input"]
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                    prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
                    prompts.append(prompt)
            else:
                prompts = []
                for i in range(len(task_examples)):
                    if args.orig_examples_percentage < 1.0:
                        if i in shuffle_indices:
                            prompt = task_prompt_sws.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        elif i in remove_stop_indices:
                            prompt = task_prompt_rsws.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        elif i in replace_words_indices:
                            prompt = task_prompt_repws.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        elif i in insert_words_indices:
                            prompt = task_prompt_insws.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        elif i in introduce_typos_indices:
                            prompt = task_prompt_intypos.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        elif i in delete_words_indices:
                            prompt = task_prompt_dws.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                        else:
                            prompt = task_prompt.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                    else:
                        prompt = task_prompt.strip() + "\n\nQ: " + task_examples[i]["input"] + "\nA:"
                    prompts.append(prompt)

            for index in random.sample(range(len(prompts)), 2):
                print(f"Sample {index} of the prompts: {prompts[index]}")

            # generate with vllm
            if args.use_vllm:
                stop = args.additional_stop_sequence
                if not args.use_chat_format or args.stop_at_double_newline:
                    stop += ["\n\n"]
                sampling_params = vllm.SamplingParams(
                    temperature=0.001,
                    max_tokens=512,
                    stop=stop,
                )
                # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
                generations = model.generate(prompts, sampling_params)
                prompt_to_output = {
                    g.prompt: g.outputs[0].text for g in generations
                }
                outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
            # generate with hf model
            else:
                stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] # get the last token because the tokenizer may add space tokens at the start.
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=512,
                    temperature=0.001,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    stop_id_sequences=[[stop_sequence] + [tokenizer.encode(stop, add_special_tokens=False) for stop in args.additional_stop_sequence]],
                )
        else:
            instances = []
            for i, example in enumerate(task_examples):
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
                instances.append({
                    "id": example["id"] if "id" in example else i,
                    "prompt": prompt,
                })
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                batch_size=args.eval_batch_size if args.eval_batch_size else 10,
                output_path=os.path.join(args.save_dir, "predictions", f"{task_name}_openai_prediction_cache.jsonl"),
            )
            outputs = [result["output"] for result in results]

        targets = [example["target"] for example in task_examples]
        predictions = []
        for example, output in zip(task_examples, outputs):
            example["raw_output"] = output
            
            # extract the first answer after `the answer is` and before the next period.
            # if there is no such answer, we will just use the raw output.
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)
            if extracted_answer:
                example["prediction"] = extracted_answer.group(1).strip()
            else:
                example["prediction"] = output.strip()
            predictions.append(example["prediction"])
        
        with open(os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in task_examples:
                fout.write(json.dumps(example) + "\n")        

        assert len(predictions) == len(targets), "number of predictions and targets are not the same."
        performance[task_name] = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]

        print(f"Task {task_name} - EM: {performance[task_name]}")

    # save the performance
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = performance
        task_name = "oi_bbh_cot"
        primary_score = results["average_exact_match"]
        upload_results_to_hf(
            results,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/bbh"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/bbh"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, 
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="if specified, chain of thoughts will be removed from the prompts."
    )
    parser.add_argument(
        "--max_num_examples_per_task", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate per task."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="load model in 4bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        '--stop_at_double_newline',
        action="store_true",
        help="If given, we will stop generation at the first double newline. Turn on to match older eval settings."
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    parser.add_argument(
        "--orig_examples_percentage", 
        type = float,
        default=1.0,
        help="The percentage of examples to keep as original."
    )
    parser.add_argument(
        "--random_seed", 
        type = int,
        default=42,
        help="The random seed."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
