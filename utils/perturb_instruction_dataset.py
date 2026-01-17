import json
import argparse
import os
from tqdm import tqdm

from instruction_perturbation import (
    extract_instruction,
    shuffle_words,
    remove_stop_words,
    delete_words,
    introduce_typos,
    insert_words_using_bert,
    replace_words_using_bert,
)

import torch
from transformers import BertTokenizer, BertForMaskedLM


def perturb_dataset(args):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    perturbation_strategies = {
        'shuffle_words':shuffle_words,
        'remove_stop_words':remove_stop_words,
        'delete_words':delete_words,
        'introduce_typos':introduce_typos,
        'insert_words_using_bert':insert_words_using_bert,
        'replace_words_using_bert':replace_words_using_bert,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    for perturbation_strategy in perturbation_strategies:
        out_file_name = 'SNI_Dolly_GPT4alpaca_' + perturbation_strategy + '.jsonl'
        output_file = os.path.join(args.output_dir, out_file_name)

        with open(output_file, 'w', encoding='utf-8') as out_perturbed:
            with open(args.input_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                for line in tqdm(lines, desc=f"perturbing instruction dataset using {perturbation_strategy} strategy"):
                    example = json.loads(line)
                    
                    messages = example["messages"]
                    user_content = ''
                    for j, message in enumerate(messages):
                        if message["role"] == "user":
                            user_content = message["content"]
                            break
                            
                    # first we separate the instruction text from the remaining user content        
                    instruc_text, remain_content = extract_instruction(user_content)
                    
                    if 'bert' in perturbation_strategy:
                        perturbed_instruc = perturbation_strategies[perturbation_strategy](instruc_text, model=model, tokenizer=tokenizer)
                    else:
                        perturbed_instruc = perturbation_strategies[perturbation_strategy](instruc_text)
                    perturbed_user_content = perturbed_instruc + '\n' + remain_content

                    example["messages"][j]["content"] = perturbed_user_content

                    out_perturbed.write(json.dumps(example) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Apply instruction perturbation.")
    parser.add_argument("--input_file", help="Path to the combined unperturbed instruction dataset file")
    parser.add_argument("--output_dir", help="Directory to save the output perturbed dataset files")
    args = parser.parse_args()
    
    perturb_dataset(args)

if __name__ == "__main__":
    main()
