
import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def replace_words_using_bert(text, model=None, tokenizer=None, percentage=0.25):
    # adapted from https://github.com/jiashenggu/Tk-Instruct/blob/main/src/ni_dataset_perturb.py
    
    words = word_tokenize(text)  # Tokenize the text
    
    # Select random words to replace (avoiding punctuation)
    words_no_punct = [w for w in words if w.isalpha()]
    num_to_replace = max(0, int(len(words_no_punct) * percentage))  # Calculate number of words to replace
    words_to_replace = random.sample(words_no_punct, num_to_replace)
    
    new_text = []
    for word in words:
        if word in words_to_replace:
            new_word = '[MASK]'
            new_text.append(new_word)  # Fallback to original if no word found
        else:
            new_text.append(word)
    
    masked_text = " ".join(new_text)
    inputs = tokenizer(masked_text, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    outputs = model(**inputs)
    predictions = outputs[0]

    _, sorted_idx = predictions[0].sort(dim=-1, descending=True)

    predicted_index = [sorted_idx[i, 0].item() for i in range(0, len(predictions[0])-1)]
    for x in range(1, len(predictions[0])-1):
        if input_ids[x] == 103:
            input_ids[x] = predicted_index[x]

    
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def insert_words_using_bert(text, model=None, tokenizer=None, percentage=0.25):
    # adapted from https://github.com/jiashenggu/Tk-Instruct/blob/main/src/ni_dataset_perturb.py

    words = word_tokenize(text)
    num_masks = max(1, int(len(words) * percentage))  # 25% of words to be added as [MASK]
    mask_indices = random.sample(range(len(words) + num_masks), num_masks)
    
    masked_words = []
    for i in range(len(words) + num_masks):
        if i in mask_indices:
            masked_words.append("[MASK]")
        if i < len(words):
            masked_words.append(words[i])
    
    masked_text = " ".join(masked_words)
    # Tokenize and prepare input
    inputs = tokenizer(masked_text, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    outputs = model(**inputs)
    predictions = outputs[0]
    
    _, sorted_idx = predictions[0].sort(dim=-1, descending=True)

    predicted_index = [sorted_idx[i, 0].item() for i in range(0, len(predictions[0])-1)]
    for x in range(1, len(predictions[0])-1):
        if input_ids[x] == 103:
            input_ids[x] = predicted_index[x]

    
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def misspell_words(word):
    if len(word) < 3:
        return word  # Avoid modifying very short words
    swap_rand_pos = random.randint(1, len(word)-2)
    ins_rand_pos = random.randint(1, len(word)-1)
    misspellings = [
        lambda w: w.replace(random.choice(w), ''),  # Remove a random letter
        lambda w: w[:swap_rand_pos] + w[swap_rand_pos:],  # Swap two letters
        lambda w: w + random.choice('aeiou'),  # Add a random vowel at the end
        lambda w: w[:ins_rand_pos] + random.choice('aeiou') + w[ins_rand_pos:],  # Insert a vowel in a random position
        lambda w: w.replace(random.choice(w), random.choice('abcdefghijklmnopqrstuvwxyz'))  # Replace a letter with a random one
    ]
    
    misspelled_word = random.choice(misspellings)(word)
    return misspelled_word

def introduce_typos(text, percentage=0.25):
    # Split the text into words
    words = word_tokenize(text)
    # Calculate the number of words to misspell
    num_to_misspell = max(1, int(len(words) * percentage))  # Ensure at least one word is removed
    indices_to_misspell = set(random.sample(range(len(words)), num_to_misspell))
    
    for i, word in enumerate(words):
        if i in indices_to_misspell:
            words[i] = misspell_words(word)

    return " ".join(words)

def delete_words(text, percentage=0.25):
    # Split the text into words
    words = word_tokenize(text)
    # Calculate the number of words to delete
    num_to_delete = max(1, int(len(words) * percentage))  # Ensure at least one word is removed
    indices_to_delete = set(random.sample(range(len(words)), num_to_delete))
    
    filtered_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
    return " ".join(filtered_words)

def shuffle_words(text, percentage=0.25):
    # Split the text into words
    words = word_tokenize(text)

    # Calculate the number of words to shuffle
    num_words_to_shuffle = int(len(words) * (percentage))

    # Create a list of indices and shuffle it
    indices = list(range(len(words)))
    random.shuffle(indices)

    # Select the indices of words to be shuffled
    shuffle_indices = indices[:num_words_to_shuffle]
    
    # Create a new list with shuffled words
    shuffled_words = words.copy()
    for i, idx in enumerate(shuffle_indices):
        shuffled_words[idx] = words[shuffle_indices[(i + 1) % num_words_to_shuffle]]

    # Join the shuffled words back into a string
    shuffled_text = ' '.join(shuffled_words)
    return shuffled_text

def remove_stop_words(text, percentage=1.0):
    # Download the stopwords data (only needed once)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    filtered_text = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a string
    return ' '.join(filtered_text)


def extract_instruction(text):
    splitted_text = text.split('\n')
    instruction_text = ''
    remain_user_content = ''

    if "You need to complete the following task:" in text:
        instruction_text = splitted_text[0] + '\n\n' + splitted_text[2]
        remain_user_content = '\n'.join(splitted_text[3:])

    elif "Below is an instruction that describes a task, paired with an input that provides further context." in text:
        instruction_text = splitted_text[0] + '\n\n' + splitted_text[2] + '\n' + splitted_text[3]
        remain_user_content = '\n'.join(splitted_text[4:])

    elif "Can you help with this?" in text:
        instruction_text = splitted_text[0] + '\n\n' + splitted_text[2]
        remain_user_content = '\n'.join(splitted_text[3:])

    elif "Tell me how would you respond to the following request." in text:
        instruction_text = splitted_text[0] + '\n' + splitted_text[1]
        remain_user_content = '\n'.join(splitted_text[2:])

    elif "Write a response that appropriately completes the request." in text:
        instruction_text = splitted_text[0] + '\n\n' + splitted_text[2] + '\n' + splitted_text[3]
        remain_user_content = '\n'.join(splitted_text[4:])

    else:
        instruction_text = splitted_text[0]
        remain_user_content = '\n'.join(splitted_text[1:])

    return instruction_text, remain_user_content