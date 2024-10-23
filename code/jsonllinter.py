import json
import sys

def lint_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                json.loads(line)  # Validate JSON
            except json.JSONDecodeError as e:
                print(f"Error in line {line_number}: {e} {line} ")

if __name__ == "__main__":

    file_path = "../../../new_data/translated_document.jsonl"
    lint_jsonl(file_path)
