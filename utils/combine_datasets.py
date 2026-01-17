import json

def main():
    # List of dataset JSONL files
    input_files = ['open-instruct/data/processed/super_ni/super_ni_data.jsonl',
                'open-instruct/data/processed/dolly/dolly_data.jsonl', 
                'open-instruct/data/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl']
    # Output JSONL file
    output_file = 'open-instruct/data/processed/SNI_Dolly_GPT4alpaca_processed_combined.jsonl'

    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in input_files:
            # Open each input file in read mode
            with open(file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)  # Write each line to the output file


if __name__ == "__main__":
    main()
