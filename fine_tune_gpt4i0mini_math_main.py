import openai
import os
from dotenv import load_dotenv
import json
import time
from openai import OpenAI
import numpy as np

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

openai.api_key = api_key
client = OpenAI()

train_file_path = "./data_process/gpt4_math_dataset_train.jsonl"
test_file_path = "./data_process/gpt4_math_dataset_test.jsonl"


def upload_file_via_api(file_path):
    """Uploads a JSONL file to OpenAI and returns the file ID."""
    print(f"Uploading file: {file_path}")
    response = openai.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    print(f"File uploaded successfully. File ID: {response.id}")
    return response.id


def create_fine_tuning_job(training_file_id, model, suffix):
    """Creates a fine-tuning job with the given training file ID and custom model name."""
    print("Creating fine-tuning job...")
    response = openai.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        suffix=suffix
    )
    print(f"Fine-tuning job created. Job ID: {response.id}")
    return response.id


def monitor_fine_tuning_job(job_id):
    """Monitors the fine-tuning job progress until completion."""
    print("Monitoring fine-tuning job progress...")
    while True:
        job_status = openai.fine_tuning.jobs.retrieve(job_id)
        status = job_status.status
        print(f"Fine-tuning status: {status}")
        if status in ["succeeded", "failed", "cancelled"]:
            if status == "succeeded":
                print(f"Fine-tuning succeeded! Fine-tuned model: {job_status.fine_tuned_model}")
                return job_status.fine_tuned_model
            else:
                print(f"Fine-tuning failed with status: {status}")
                return None
        time.sleep(30)


def run_fine_tuned_model(fine_tuned_model, test_file_path):
    """Tests the fine-tuned model using prompts from the test dataset."""
    print("Testing the fine-tuned model...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            user_prompt = data["messages"][1]["content"]
            response = openai.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who solves math problems step-by-step."},
                    {"role": "user", "content": user_prompt}
                ]
            )
            print(f"User Prompt: {user_prompt}")
            print(f"Response: {response.choices[0].message.content}")
            print("-" * 50)

def get_embedding(text, model="text-embedding-ada-002"):
    """Fetches an embedding vector for a given text."""
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_accuracy(fine_tuned_model, test_file_path, similarity_threshold=0.85):
    """Calculates the overall accuracy of the fine-tuned model on the test dataset."""
    correct_count = 0
    total_count = 0

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            user_prompt = data["messages"][1]["content"]
            expected_answer = data["messages"][2]["content"]

            # Call the fine-tuned model
            response = openai.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who solves math problems step-by-step."},
                    {"role": "user", "content": user_prompt}
                ]
            )
            model_response = response.choices[0].message.content.strip()

            model_embedding = get_embedding(model_response)
            expected_embedding = get_embedding(expected_answer)

            similarity = cosine_similarity(model_embedding, expected_embedding)

            if similarity >= similarity_threshold:
                correct_count += 1
            total_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

def list_models():
    client = OpenAI()
    response = client.fine_tuning.jobs.list()
    print("Available models:")
    for model in response:
        print(f"Model ID: {model}")

if __name__ == "__main__":
    training_file_id = upload_file_via_api(train_file_path)

    fine_tuning_job_id = create_fine_tuning_job(training_file_id, model="gpt-4o-mini-2024-07-18", suffix="GPT4o_MATH")

    fine_tuned_model_name = monitor_fine_tuning_job(fine_tuning_job_id)

    #fine_tuned_model_name = 'ft:gpt-3.5-turbo-0125:colubmia-university:gpt3-5-math:Aefm0Kur'

    #list_models()

    #if fine_tuned_model_name:
        #calculate_accuracy(fine_tuned_model_name, test_file_path)
    #else:
        #print("Fine-tuning was not successful.")
