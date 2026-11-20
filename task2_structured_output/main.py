
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


# Load environment variables from the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


# Pydantic model for structured ticket information
class TicketInfo(BaseModel):
    customer_name: str
    product: str
    issue_type: str
    urgency: int = Field(ge=1, le=5)
    summary: str


# Extract structured information with validation and retry
def extract_ticket(ticket_text):

    error_feedback = ""

    for attempt in range(1, 4):

        print(f"\nAttempt {attempt} of 3")

        prompt = f"""
You are an expert customer support ticket information extractor.

TASK:
Extract information from the customer support ticket below.

Return ONLY valid JSON with these exact fields:
- customer_name: string
- product: string
- issue_type: string
- urgency: integer between 1 and 5
- summary: string

EXTRACTION RULES:

1. CUSTOMER NAME:
- Extract the exact customer name whenever it appears.
- If the ticket says "My name is John", return "John".
- If the ticket says "I am David", return "David".
- If the ticket says "Customer Sarah", return "Sarah".
- Use "Unknown" only if no name is provided.

2. PRODUCT:
- Extract the product name from the ticket.
- Examples: Windows software, Microsoft Office,
  antivirus subscription, accounting software.
- Use "Unknown" only if no product is provided.

3. ISSUE TYPE:
- If the software crashes, return "Technical - Crashing".
- If the application is slow, return "Technical - Performance".
- If the customer cannot activate a product, return "Activation".
- If the customer reports a payment problem, return "Payment".
- If the customer cannot log in, return "Login".
- Identify the actual issue described in the ticket.

4. URGENCY:
- 1 = Very low
- 2 = Low
- 3 = Medium
- 4 = High
- 5 = Critical
- The urgency must be an integer, not a string.
- If the customer's work is blocked or the issue is extremely urgent,
  use 5.

5. SUMMARY:
- Write a short summary of the customer's problem.
- Do not invent information.

EXAMPLES:

Ticket: My name is John. I purchased Windows software and it keeps crashing. This is urgent.

Output:
{{
  "customer_name": "John",
  "product": "Windows software",
  "issue_type": "Technical - Crashing",
  "urgency": 5,
  "summary": "Windows software keeps crashing and the customer reports urgency."
}}

Ticket: I am David. My antivirus subscription payment was deducted twice.

Output:
{{
  "customer_name": "David",
  "product": "antivirus subscription",
  "issue_type": "Payment",
  "urgency": 4,
  "summary": "The customer reports that the antivirus subscription payment was deducted twice."
}}

CUSTOMER TICKET:
{ticket_text}

{error_feedback}

Return only the JSON object. Do not include Markdown or explanations.
"""

        try:

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
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0
                    }
                },
                timeout=120
            )

            response.raise_for_status()

            raw_json = response.json()["message"]["content"]

            # Validate model response using Pydantic
            ticket_info = TicketInfo.model_validate_json(raw_json)

            print("Validation successful!")

            print(ticket_info.model_dump_json(indent=2))

            return {
                "success": True,
                "data": ticket_info.model_dump(),
                "error": None
            }

        except ValidationError as error:

            print("Pydantic validation failed:")
            print(error)

            error_feedback = f"""
Your previous JSON response failed Pydantic validation.

Validation errors:
{error}

Previous response:
{raw_json if 'raw_json' in locals() else 'No response received'}

Correct the errors and return valid JSON with all required fields.
"""

        except (requests.RequestException, ValueError, KeyError, TypeError) as error:

            print("Extraction error:", error)

            error_feedback = f"""
Your previous attempt failed with this error:
{error}

Please try again and return valid JSON.
"""

    print("\nAll 3 attempts failed.")

    return {
        "success": False,
        "data": None,
        "error": "Extraction or validation failed after 3 attempts."
    }


# Five customer support tickets for testing
tickets = [

    "My name is John. I purchased Windows software and it keeps crashing. This is urgent.",

    "Customer Sarah purchased Microsoft Office. She cannot activate her product.",

    "I am David. My antivirus subscription payment was deducted twice.",

    "Customer Priya bought accounting software. She cannot log in to her account.",

    "My name is Robert. The application is extremely slow and my work is blocked."

]


# Main program
if __name__ == "__main__":

    print("=" * 60)
    print("TASK 2 - STRUCTURED OUTPUT WITH PYDANTIC VALIDATION")
    print("=" * 60)

    successful = 0
    failed = 0

    for index, ticket in enumerate(tickets, start=1):

        print(f"\n\nTICKET {index}")
        print("-" * 60)

        print("Customer Ticket:", ticket)

        result = extract_ticket(ticket)

        if result["success"]:

            successful += 1
            print("\nFinal Result: SUCCESS")

        else:

            failed += 1
            print("\nFinal Result: FAILED")
            print(result["error"])

    print("\n\n" + "=" * 60)
    print("TASK 2 FINAL SUMMARY")
    print("=" * 60)

    print("Total Tickets:", len(tickets))
    print("Successful Extractions:", successful)
    print("Failed Extractions:", failed)
    print("Success Rate:", f"{(successful / len(tickets)) * 100:.2f}%")
