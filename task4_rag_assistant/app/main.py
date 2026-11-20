
import os
import json
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from task4_rag_assistant.app.rag import ask_question

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

app = FastAPI(
    title="GenAI Assessment - RAG Assistant",
    description="PDF question answering, classification and combined chat",
    version="1.0"
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=2)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    customer_id: int | None = None
    order_id: int | None = None


# --------------------------------------------------
# OLLAMA HELPER
# --------------------------------------------------

def ask_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    except requests.RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama connection failed: {error}"
        )


# --------------------------------------------------
# TASK 4 - CLASSIFICATION
# --------------------------------------------------

CLASSIFICATION_PROMPT = """
You are a customer support ticket classifier.

Classify the customer message into exactly one category:
- Billing
- Technical
- Account

Rules:
- Billing: payment, invoice, refund, subscription or charges.
- Technical: software errors, bugs, crashes or performance.
- Account: login, password, profile or account access.

Return only the category name.
Do not explain your answer.
"""


@app.get("/")
def home():
    return {
        "message": "GenAI RAG Assistant is running",
        "endpoints": ["/ask", "/classify", "/chat", "/docs"]
    }


@app.post("/ask")
def ask_endpoint(request: QuestionRequest):
    try:
        result = ask_question(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/classify")
def classify_endpoint(request: QuestionRequest):
    prompt = f"""
{CLASSIFICATION_PROMPT}

Customer message:
{request.question}

Category:
"""

    answer = ask_ollama(prompt)

    allowed_categories = ["Billing", "Technical", "Account"]

    if answer not in allowed_categories:
        answer = "Unknown"

    return {
        "question": request.question,
        "category": answer
    }


# --------------------------------------------------
# TASK 3 TOOL FUNCTIONS
# --------------------------------------------------

def get_customer(customer_id):
    customers = {
        1: {
            "customer_id": 1,
            "name": "John",
            "plan": "Basic"
        },
        2: {
            "customer_id": 2,
            "name": "Sarah",
            "plan": "Premium"
        },
        3: {
            "customer_id": 3,
            "name": "David",
            "plan": "Enterprise"
        }
    }

    return customers.get(customer_id)



def get_order_status(order_id):
    import sqlite3

    database_path = os.path.join(
        ROOT_DIR,
        "task3_tool_calling",
        "orders.db"
    )

    if not os.path.exists(database_path):
        return None

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        # Read the actual database columns
        cursor.execute("PRAGMA table_info(orders)")
        columns = [
            row["name"] for row in cursor.fetchall()
        ]

        # Retrieve the order without assuming an amount column
        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,)
        )

        result = cursor.fetchone()

        if result is None:
            return None

        return dict(result)

    finally:
        connection.close()



# --------------------------------------------------
# TASK 4 - COMBINED CHAT
# --------------------------------------------------

@app.post("/chat")
def chat_endpoint(request: ChatRequest):

    customer_information = None
    order_information = None

    if request.customer_id is not None:
        customer_information = get_customer(
            request.customer_id
        )

    if request.order_id is not None:
        order_information = get_order_status(
            request.order_id
        )

    rag_result = ask_question(request.question)

    prompt = f"""
You are a customer support assistant.

Answer using ONLY the information provided below.
Do not invent customer details, order details, product features,
prices, delivery dates or policies.

CUSTOMER INFORMATION:
{json.dumps(customer_information)}

ORDER INFORMATION:
{json.dumps(order_information)}

PRODUCT MANUAL INFORMATION:
{rag_result["answer"]}

CUSTOMER QUESTION:
{request.question}

If the information is unavailable, clearly say:
"I don't know."

Give a helpful and concise answer.
"""

    answer = ask_ollama(prompt)

    return {
        "question": request.question,
        "answer": answer,
        "sources": rag_result["sources"],
        "customer": customer_information,
        "order": order_information
    }
