import json
import argparse
import os
from tqdm import tqdm
import random
import re

random.seed(42)

def get_balanced_perturbation(args, shuffled_data, num_examples):
    # Dictionary to track the last transformation used for each exmple
    last_perturbations = [set() for _ in range(num_examples)]

    # Track which examples have been selected to keep with no perturbation
    original_selected = set()
    num_original_examples = num_examples * int(args.percentage_of_original) // 100
    num_perturbed_examples = num_examples * (100 - int(args.percentage_of_original)) // 100
    print('num_original_examples', num_original_examples)
    print('num_perturbed_examples', num_perturbed_examples)


    indices = list(range(num_examples))
    epochs = {}

    for epoch in range(int(args.num_epochs)):
        print('epoch', epoch)
        epoch_data = {}
        available_indices = indices.copy()
        random.shuffle(available_indices)
        
        # Select unique examples to keep as original that haven't been selected before
        remaining_choices = list(set(available_indices) - original_selected)
        if len(remaining_choices) > 0 and len(remaining_choices) < num_original_examples:
            for i in available_indices:
                if i not in remaining_choices:
                    remaining_choices.append(i)
                if len(remaining_choices) == num_original_examples:
                    break
        if len(remaining_choices) == 0:
            original_selected.clear()
            remaining_choices = available_indices.copy()
        original_indices = random.sample(remaining_choices, num_original_examples)
        original_selected.update(original_indices)

        for idx in tqdm(original_indices, desc="Adding original, unperturbed, examples"):
            epoch_data[idx] = shuffled_data["original"][idx]
            available_indices.remove(idx)
        
        perturbation_methods = [file_name for file_name in shuffled_data.keys() if file_name != "original"]
        counter = {perturbation_method: 0 for perturbation_method in perturbation_methods}
        for idx in tqdm(available_indices, desc="Adding perturbed examples"):
            valid_perturbations = [perturb_method for perturb_method in perturbation_methods if perturb_method not in list(last_perturbations[idx]) and counter[perturb_method] < (num_perturbed_examples // 6)]
            # If all valid options are exhausted, pick any from the remaining pool
            if not valid_perturbations:
                valid_perturbations = perturbation_methods
            
            chosen_perturbation = random.choice(valid_perturbations)
            
            epoch_data[idx] = shuffled_data[chosen_perturbation][idx]
            last_perturbations[idx].add(chosen_perturbation)
            counter[chosen_perturbation] += 1
            
        # shuffle the epoch data
        keys = list(epoch_data.keys())  # Get dictionary keys
        random.shuffle(keys)   # Shuffle the keys
        epoch_data_shuffled = {key: epoch_data[key] for key in keys}  # Reconstruct the dictionary
        epochs[epoch] = epoch_data_shuffled
    # print(epochs)
    return epochs


def create_dataset(args):
    print(f'*** Generating a training epochs dataset using {args.num_epochs} epochs and keeping {args.percentage_of_original} percent of the instructions unperturbed. ***')
    all_data = {}
    counter = 0
    for file in os.listdir(args.perturbed_dir):
        if file.endswith(".jsonl"):
            pattern = r'SNI_Dolly_GPT4alpaca_(.*)\.jsonl'
            file_name = re.search(pattern, file).group(1)
            print(file_name)
            with open(os.path.join(args.perturbed_dir, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                example_list = []
                for line in tqdm(lines, desc="Reading perturbed instruction dataset"):
                    example = json.loads(line)
                    example["perturbed"] = True
                    example["perturbation_method"] = file_name
                    example_list.append(example)
                all_data[file_name] = example_list

    # read original dataset
    with open(args.original_dataset, "r", encoding="utf-8") as f:
        lines = f.readlines()
        example_list = []
        for line in tqdm(lines, desc="Reading original instruction dataset"):
            example = json.loads(line)
            example["perturbed"] = False
            example["perturbation_method"] = None
            example_list.append(example)
        all_data["original"] = example_list

    # Ensure all files have the same number of examples
    num_examples = len(next(iter(all_data.values())))
    assert all(len(data) == num_examples for data in all_data.values()), "Files have different number of lines!"
    print(num_examples)

    # Generate a shuffled index order
    shuffled_indices = list(range(num_examples))
    random.shuffle(shuffled_indices)

    # Apply the same shuffle order to each file to keep the same order of examples accross perturbed datasets
    shuffled_data = {filename: [data[i] for i in shuffled_indices] for filename, data in all_data.items()}
    
    training_epochs = get_balanced_perturbation(args, shuffled_data, num_examples)

    os.makedirs(args.output_dir, exist_ok=True)
    out_file_name = f'SNI_Dolly_GPT4alpaca_perturbed_training_epochs_with_{args.percentage_of_original}_perc_original_instructions.jsonl'
    output_file = os.path.join(args.output_dir, out_file_name)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for epoch in tqdm(training_epochs.values(), desc="Writing training dataset epochs"):
            for example in epoch.values():
                out_file.write(json.dumps(example) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Apply instruction perturbation.")
    parser.add_argument("--perturbed_dir", help="Path to the directory of the perturbed and orginial dataset files")
    parser.add_argument("--original_dataset", help="Path to the orginial dataset file")
    parser.add_argument("--output_dir", help="Directory to save the output perturbed dataset files")
    parser.add_argument("--percentage_of_original", default=100, help="The percentage of original unperturbed instructions to keep")
    parser.add_argument("--num_epochs", default=1, help="The number of training epochs")
    args = parser.parse_args()
    
    create_dataset(args)

if __name__ == "__main__":
    main()
