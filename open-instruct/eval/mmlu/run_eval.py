import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from categories import subcategories, categories

from transformers import BertTokenizer, BertForMaskedLM

import sys
# getting the name of the directory
# where this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata
# from eval import utils

import random
from utils import (
    shuffle_words, 
    remove_stop_words,
    replace_words_using_bert,
    insert_words_using_bert,
    introduce_typos,
    delete_words,
)

choices = ["A", "B", "C", "D"]

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

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True, perturbation_strategy=None, random_seed=42):
    prompt = df.iloc[idx, 0]

    # perturb the instruction prompt
    if perturbation_strategy:
        if 'bert' in perturbation_strategy:
            prompt = perturbation_func[perturbation_strategy](prompt, model=bert_model, tokenizer=bert_tokenizer, random_seed=random_seed)
        else:
            prompt = perturbation_func[perturbation_strategy](prompt, random_seed=random_seed)

    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1, perturbation_strategy=None, random_seed=42):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    # perturb the instruction prompt
    if perturbation_strategy:
        if 'bert' in perturbation_strategy:
            prompt = perturbation_func[perturbation_strategy](prompt, model=bert_model, tokenizer=bert_tokenizer, random_seed=random_seed)
        else:
            prompt = perturbation_func[perturbation_strategy](prompt, random_seed=random_seed)


    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, perturbation_strategy=perturbation_strategy, random_seed=random_seed)
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    
    n_perturbation = 0
    if args.orig_examples_percentage < 1.0:
        # Apply mix perturbation
        n_examples = test_df.shape[0]
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
    
    for i in range(0, test_df.shape[0]):
        perturbation_strategy = None
        if n_perturbation > 0:
            if i in shuffle_indices:
                perturbation_strategy = "shuffle_words"
            elif i in remove_stop_indices:
                perturbation_strategy = "remove_stop_words"
            elif i in replace_words_indices:
                perturbation_strategy = "replace_words_using_bert"
            elif i in insert_words_indices:
                perturbation_strategy = "insert_words_using_bert"
            elif i in introduce_typos_indices:
                perturbation_strategy = "introduce_typos"
            elif i in delete_words_indices:
                perturbation_strategy = "delete_words"
            
        
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False, perturbation_strategy=perturbation_strategy, random_seed=args.random_seed)
        train_prompt = gen_prompt(dev_df, subject, k, perturbation_strategy=perturbation_strategy, random_seed=args.random_seed)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"
                    
            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    # Log a few random samples from the prompts:
    for index in random.sample(range(len(prompts)), 3):
        print(f"Sample {index} of the prompts: {prompts[index]}")

    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end        
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )
    
    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]) # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def main(args):
    random.seed(args.random_seed)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
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
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    if args.orig_examples_percentage < 1.0:
        print(f"Perturbing the instruction text using mix perturbation strategy. {args.orig_examples_percentage * 100}% of examples are kept as original with no perturbation. {(1 - args.orig_examples_percentage) * 100}% of examples are splitted equally between the 6 different perurbation strategies.")

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size)
            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat]))
                    for cat in cat_cors
                },
            },
            f,
        )

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = {
            "average_acc": weighted_acc,
            "subcat_acc": {
                subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                for subcat in subcat_cors
            },
            "cat_acc": {
                cat: np.mean(np.concatenate(cat_cors[cat]))
                for cat in cat_cors
            },
        }
        task_name = f"oi_mmlu_{args.ntrain}shots"
        primary_score = results["average_acc"]
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
        "--ntrain",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
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
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
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
