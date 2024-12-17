import os
import json

def process_math_data(root_dir):
    """
    Processes math problems in a structured directory and converts them to GPT-4 fine-tuning JSONL format.
    """
    data_output = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.endswith(".json") or file_name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        try:
                            data = json.load(file)
                        except:
                            data = {"problem": file.readline(), "solution": file.read()}

                    problem = data.get("problem", "No problem provided")
                    solution = data.get("solution", "No solution provided")

                    entry = {
                        "messages": [
                            {"role": "system",
                             "content": "You are a helpful assistant who solves math problems step-by-step."},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution}
                        ]
                    }
                    data_output.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Data processing complete! Output written to {output_file}")

root_dir = "../MATH/train/"
output_file = "gpt4_math_dataset_train.jsonl"
process_math_data(root_dir)
root_dir = "../MATH/test/"
output_file = "gpt4_math_dataset_test.jsonl"
process_math_data(root_dir)