import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM

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
    generate_completions,
    load_hf_lm,
    query_openai_chat_model,
    dynamic_import_function,
    load_hf_tokenizer,
    upload_results_to_hf,
    check_and_upload_model_metadata
)
from examplars import EXAMPLARS as GSM_EXAMPLARS
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

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    if args.orig_examples_percentage < 1.0:
        print(f"Perturbing the instruction text using mix perturbation strategy. {args.orig_examples_percentage * 100}% of examples are kept as original with no perturbation. {(1 - args.orig_examples_percentage) * 100}% of examples are splitted equally between the 6 different perurbation strategies.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    global GSM_EXAMPLARS
    if args.n_shot:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        demonstrations_sws = []
        demonstrations_rsws = []
        demonstrations_repws = []
        demonstrations_insws = []
        demonstrations_intypos = []
        demonstrations_dws = []
        for example in GSM_EXAMPLARS:
            if args.orig_examples_percentage < 1.0:
                question_sws = perturbation_func["shuffle_words"](example["question"], random_seed=args.random_seed)
                question_rsws = perturbation_func["remove_stop_words"](example["question"], random_seed=args.random_seed)
                question_repws = perturbation_func["replace_words_using_bert"](example["question"], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                question_insws = perturbation_func["insert_words_using_bert"](example["question"], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                question_intypos = perturbation_func["introduce_typos"](example["question"], random_seed=args.random_seed)
                question_dws = perturbation_func["delete_words"](example["question"], random_seed=args.random_seed)
            if args.no_cot:
                demonstrations.append(
                    "Question: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                )
                if args.orig_examples_percentage < 1.0:
                    demonstrations_sws.append(
                        "Question: " + question_sws + "\n" + "Answer: " + example["short_answer"]
                    )
                    demonstrations_rsws.append(
                        "Question: " + question_rsws + "\n" + "Answer: " + example["short_answer"]
                    )
                    demonstrations_repws.append(
                        "Question: " + question_repws + "\n" + "Answer: " + example["short_answer"]
                    )
                    demonstrations_insws.append(
                        "Question: " + question_insws + "\n" + "Answer: " + example["short_answer"]
                    )
                    demonstrations_intypos.append(
                        "Question: " + question_intypos + "\n" + "Answer: " + example["short_answer"]
                    )
                    demonstrations_dws.append(
                        "Question: " + question_dws + "\n" + "Answer: " + example["short_answer"]
                    )
            else:
                demonstrations.append(
                    "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                )
                if args.orig_examples_percentage < 1.0:
                    demonstrations_sws.append(
                        "Question: " + question_sws + "\n" + "Answer: " + example["cot_answer"]
                    )
                    demonstrations_rsws.append(
                        "Question: " + question_rsws + "\n" + "Answer: " + example["cot_answer"]
                    )
                    demonstrations_repws.append(
                        "Question: " + question_repws + "\n" + "Answer: " + example["cot_answer"]
                    )
                    demonstrations_insws.append(
                        "Question: " + question_insws + "\n" + "Answer: " + example["cot_answer"]
                    )
                    demonstrations_intypos.append(
                        "Question: " + question_intypos + "\n" + "Answer: " + example["cot_answer"]
                    )
                    demonstrations_dws.append(
                        "Question: " + question_dws + "\n" + "Answer: " + example["cot_answer"]
                    )

        prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        if args.orig_examples_percentage < 1.0:
            prompt_prefix_sws = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_sws) + "\n\n"
            prompt_prefix_rsws = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_rsws) + "\n\n"
            prompt_prefix_repws = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_repws) + "\n\n"
            prompt_prefix_insws = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_insws) + "\n\n"
            prompt_prefix_intypos = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_intypos) + "\n\n"
            prompt_prefix_dws = "Answer the following questions.\n\n" + "\n\n".join(demonstrations_dws) + "\n\n"
    else:
        prompt_prefix = "Answer the following question.\n\n"
        if args.orig_examples_percentage < 1.0:
            prompt_prefix_sws = "Answer the following question.\n\n"
            prompt_prefix_rsws = "Answer the following question.\n\n"
            prompt_prefix_repws = "Answer the following question.\n\n"
            prompt_prefix_insws = "Answer the following question.\n\n"
            prompt_prefix_intypos = "Answer the following question.\n\n"
            prompt_prefix_dws = "Answer the following question.\n\n"

    if args.use_chat_format:
        chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
        def apply_chat_format(example, tokenizer):
            messages = [{"role": "user", "content": prompt_prefix + "Question: " + example["question"].strip()}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
            return prompt

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if args.use_vllm:
            if args.load_in_4bit:
                model = vllm.LLM(
                    model=args.model_name_or_path,
                    quantization="bitsandbytes",
                    load_format= "bitsandbytes",
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
            stop_strings = args.additional_stop_sequence
            # we only use stop token for non-chat format (usually applied to vanilla pretrained language models).
            # For chat format, we will rely on the model knows when to stop.
            if not args.use_chat_format:
                stop_strings += ["\n\n"] if args.stop_at_double_newline else ["\n"]
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=512,
                stop=stop_strings
            )
            if args.use_chat_format:
                prompts = [apply_chat_format(example, tokenizer) for example in test_data]
            else:
                if args.orig_examples_percentage < 1.0:
                    n_examples = len(test_data)
                    # Calculate number of examples for each perturbation
                    n_perturbation = int(n_examples * (1 - args.orig_examples_percentage)/6) # we have 6 perturbation methods
        
                    
                    # Randomly select indices for modifications
                    all_indices = list(range(n_examples))
                    random.shuffle(all_indices)
                    
                    # Get indices for each modification
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

                prompts = []
                for i in range(len(test_data)):
                    if args.orig_examples_percentage < 1.0:
                        if i in shuffle_indices:
                            question_sws = perturbation_func["shuffle_words"](test_data[i]["question"], random_seed=args.random_seed)
                            prompt = prompt_prefix_sws + "Question: " + question_sws.strip() + "\nAnswer:"
                        elif i in remove_stop_indices:
                            question_rsws = perturbation_func["remove_stop_words"](test_data[i]["question"], random_seed=args.random_seed)
                            prompt = prompt_prefix_rsws + "Question: " + question_rsws.strip() + "\nAnswer:"
                        elif i in replace_words_indices:
                            question_repws = perturbation_func["replace_words_using_bert"](test_data[i]["question"], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                            prompt = prompt_prefix_repws + "Question: " + question_repws.strip() + "\nAnswer:"
                        elif i in insert_words_indices:
                            question_insws = perturbation_func["insert_words_using_bert"](test_data[i]["question"], model=bert_model, tokenizer=bert_tokenizer, random_seed=args.random_seed)
                            prompt = prompt_prefix_insws + "Question: " + question_insws.strip() + "\nAnswer:"
                        elif i in introduce_typos_indices:
                            question_intypos = perturbation_func["introduce_typos"](test_data[i]["question"], random_seed=args.random_seed)
                            prompt = prompt_prefix_intypos + "Question: " + question_intypos.strip() + "\nAnswer:"
                        elif i in delete_words_indices:
                            question_dws = perturbation_func["delete_words"](test_data[i]["question"], random_seed=args.random_seed)
                            prompt = prompt_prefix_dws + "Question: " + question_dws.strip() + "\nAnswer:"
                        else:
                            prompt = prompt_prefix + "Question: " + test_data[i]["question"].strip() + "\nAnswer:"
                    else:
                        prompt = prompt_prefix + "Question: " + test_data[i]["question"].strip() + "\nAnswer:"
                    prompts.append(prompt)
                # prompts = [prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:" for example in test_data]
            
            for index in random.sample(range(len(prompts)), 3):
                print(f"Sample {index} of the prompts: {prompts[index]}")

            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                revision=args.hf_revision,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
            if args.use_chat_format:
                prompts = [apply_chat_format(example, tokenizer) for example in test_data]
            else:
                prompts = [prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer:" for example in test_data]            
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            stop_tokens = [[new_line_token]]
            stop_tokens += [[tokenizer.encode(stop_seq, add_special_tokens=False)[-1]] for stop_seq in args.additional_stop_sequence]
            if args.stop_at_double_newline:
                # We'll stop generation at double new line (check if that's 1 or 2 tokens)
                double_new_line_token = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
                if new_line_token == double_new_line_token:
                    stop_tokens = [new_line_token, new_line_token]   # double new line is two new line tokens
                else:
                    stop_tokens = [double_new_line_token]  # double new line has its own token
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[stop_tokens] if not args.use_chat_format else None,  # we only use stop token for non-chat format (usually applied to vanilla pretrained language models). For chat format, we will rely on the model knows when to stop.
                do_sample=False,
            )
    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
        
    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = { "exact_match": em_score }
        task_name = "oi_gsm8k_cot"
        primary_score = results["exact_match"]
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
        default="data/gsm"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/gsm"
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
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
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
        "--stop_at_double_newline",
        action="store_true",
        help="If given, will stop generation at double newline instead of single."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
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
