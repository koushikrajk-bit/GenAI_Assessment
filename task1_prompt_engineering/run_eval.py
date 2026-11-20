
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATASET_FILE = BASE_DIR / "golden.jsonl"


def load_prompt(version):
    prompt_file = PROMPTS_DIR / f"classifier_{version}.txt"

    with open(prompt_file, "r", encoding="utf-8") as file:
        return file.read()


def classify_ticket(ticket, version):
    prompt = load_prompt(version).replace("{ticket}", ticket)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()["message"]["content"].strip()

    # V2 returns JSON; V1 returns plain category text
    if version == "v2":
        try:
            result = json.loads(result)["category"]
        except (json.JSONDecodeError, KeyError, TypeError):
            result = "INVALID"

    result = result.strip().strip('"').strip("'")

    if result not in ["Billing", "Technical", "Account"]:
        return "INVALID"

    return result


def load_dataset():
    tickets = []

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                tickets.append(json.loads(line))

    return tickets


def run_eval(version):
    dataset = load_dataset()
    correct = 0

    print(f"\nEvaluating Classifier {version.upper()}")
    print("-" * 40)

    for index, item in enumerate(dataset, start=1):
        expected = item["category"]

        try:
            predicted = classify_ticket(
                item["ticket"],
                version
            )
        except Exception as error:
            predicted = "ERROR"
            print(f"Ticket {index}: API error - {error}")

        if predicted == expected:
            correct += 1

        print(
            f"Ticket {index}: "
            f"Expected={expected}, Predicted={predicted}"
        )

    accuracy = (correct / len(dataset)) * 100

    print("-" * 40)
    print(f"Correct: {correct}/{len(dataset)}")
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    v1_accuracy = run_eval("v1")
    v2_accuracy = run_eval("v2")

    print("\nFINAL COMPARISON")
    print(f"V1 Accuracy: {v1_accuracy:.2f}%")
    print(f"V2 Accuracy: {v2_accuracy:.2f}%")
    print(f"Improvement: {v2_accuracy - v1_accuracy:.2f} percentage points")
