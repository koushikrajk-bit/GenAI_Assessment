
import os
import json
import sqlite3
import requests
from pathlib import Path
from dotenv import load_dotenv


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

DATABASE_PATH = Path(__file__).resolve().parent / "orders.db"


# ==================================================
# 2. MOCK CRM CUSTOMER DATA
# ==================================================

CUSTOMERS = {
    1: {
        "customer_id": 1,
        "name": "John",
        "email": "john@example.com",
        "plan": "Basic"
    },
    2: {
        "customer_id": 2,
        "name": "Sarah",
        "email": "sarah@example.com",
        "plan": "Premium"
    },
    3: {
        "customer_id": 3,
        "name": "David",
        "email": "david@example.com",
        "plan": "Enterprise"
    }
}


# ==================================================
# 3. INITIALIZE SQLITE DATABASE
# ==================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL
        )
    """)

    sample_orders = [
        (1042, 2, "Laptop", "Shipped", 75000.00),
        (1043, 1, "Keyboard", "Processing", 2500.00),
        (1044, 3, "Monitor", "Delivered", 18000.00),
        (1045, 2, "Mouse", "Delivered", 1200.00)
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO orders
        (order_id, customer_id, product, status, total_amount)
        VALUES (?, ?, ?, ?, ?)
    """, sample_orders)

    connection.commit()
    connection.close()

    print("SQLite database initialized successfully.")


# ==================================================
# 4. TOOL 1 - GET CUSTOMER DETAILS
# ==================================================

def get_customer(customer_id):

    try:
        customer_id = int(customer_id)

    except (ValueError, TypeError):
        return {
            "success": False,
            "error": "Invalid customer ID."
        }

    customer = CUSTOMERS.get(customer_id)

    if customer is None:
        return {
            "success": False,
            "error": f"Customer ID {customer_id} was not found."
        }

    return {
        "success": True,
        "customer": customer
    }


# ==================================================
# 5. TOOL 2 - GET ORDER STATUS
# ==================================================

def get_order_status(order_id):

    try:
        order_id = int(order_id)

    except (ValueError, TypeError):
        return {
            "success": False,
            "error": "Invalid order ID."
        }

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT order_id, customer_id, product, status, total_amount
        FROM orders
        WHERE order_id = ?
    """, (order_id,))

    order = cursor.fetchone()

    connection.close()

    if order is None:
        return {
            "success": False,
            "error": f"Order ID {order_id} was not found."
        }

    return {
        "success": True,
        "order": dict(order)
    }


# ==================================================
# 6. DEFINE AVAILABLE TOOLS
# ==================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": (
                "Get customer details including name, email "
                "and subscription plan using customer ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer ID"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Get order details including product, status "
                "and total amount using order ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]


# ==================================================
# 7. EXECUTE TOOL SAFELY
# ==================================================

def execute_tool(tool_name, arguments):

    if not isinstance(arguments, dict):

        return {
            "success": False,
            "error": "Tool arguments must be a JSON object."
        }

    if tool_name == "get_customer":

        if "customer_id" not in arguments:

            return {
                "success": False,
                "error": "Missing customer_id argument."
            }

        return get_customer(arguments["customer_id"])

    elif tool_name == "get_order_status":

        if "order_id" not in arguments:

            return {
                "success": False,
                "error": "Missing order_id argument."
            }

        return get_order_status(arguments["order_id"])

    else:

        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }


# ==================================================
# 8. MODEL TOOL CALLING
# ==================================================

def ask_assistant(question):

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful customer support assistant.

You have access to two tools:

1. get_customer:
   Use this tool when the user asks about customer details
   or subscription plans.

2. get_order_status:
   Use this tool when the user asks about orders or delivery status.

IMPORTANT RULES:

- Decide which tool or tools are needed based on the question.
- You may call both tools for one question.
- Never invent customer or order information.
- Only report information explicitly returned by the tools.
- Do not assume or invent a delivery location.
- If the database only says "Shipped", report only "Shipped".
- If a tool reports that an ID was not found, clearly tell the user.
- Do not claim a tool succeeded when it returned an error.
- Do not add facts that are not present in the tool results.
- Give a clear and concise answer based on the tool results.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Maximum five tool-calling rounds
    for round_number in range(5):

        try:

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": TOOLS,
                    "stream": False,
                    "options": {
                        "temperature": 0
                    }
                },
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            assistant_message = result.get("message", {})

            tool_calls = assistant_message.get("tool_calls", [])

            # Return final answer if no more tools are requested
            if not tool_calls:

                answer = assistant_message.get("content", "")

                return {
                    "success": True,
                    "answer": answer
                }

            # Add assistant tool-call message
            messages.append(assistant_message)

            # Execute all requested tools
            for tool_call in tool_calls:

                function_data = tool_call.get("function", {})

                tool_name = function_data.get("name", "")

                arguments = function_data.get("arguments", {})

                print("\nModel requested tool:", tool_name)

                print("Arguments:", arguments)

                tool_result = execute_tool(
                    tool_name,
                    arguments
                )

                print("\nTool result:")

                print(json.dumps(tool_result, indent=2))

                # Send tool result back to the model
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result)
                })

        except requests.RequestException as error:

            return {
                "success": False,
                "answer": None,
                "error": f"Ollama request failed: {error}"
            }

        except (ValueError, KeyError, TypeError) as error:

            return {
                "success": False,
                "answer": None,
                "error": f"Invalid model response: {error}"
            }

    return {
        "success": False,
        "answer": None,
        "error": "Maximum tool-calling rounds reached."
    }


# ==================================================
# 9. DIRECT TOOL TESTS
# ==================================================

def test_tools():

    print("\n" + "=" * 60)
    print("DIRECT TOOL TESTS")
    print("=" * 60)

    print("\nTEST 1 - Existing customer")

    print(json.dumps(
        get_customer(2),
        indent=2
    ))

    print("\nTEST 2 - Unknown customer")

    print(json.dumps(
        get_customer(999),
        indent=2
    ))

    print("\nTEST 3 - Existing order")

    print(json.dumps(
        get_order_status(1042),
        indent=2
    ))

    print("\nTEST 4 - Nonexistent order")

    print(json.dumps(
        get_order_status(9999),
        indent=2
    ))

    print("\nTEST 5 - Unknown tool")

    print(json.dumps(
        execute_tool(
            "delete_customer",
            {"customer_id": 2}
        ),
        indent=2
    ))


# ==================================================
# 10. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 3 - TOOL CALLING")
    print("=" * 60)

    # Create database and insert sample orders
    initialize_database()

    # Test tools independently
    test_tools()

    # Test model-driven tool calling
    print("\n" + "=" * 60)
    print("MODEL TOOL-CALLING TEST")
    print("=" * 60)

    question = (
        "Which plan is customer 2 on "
        "and where is order 1042?"
    )

    print("\nUser question:", question)

    result = ask_assistant(question)

    print("\n" + "=" * 60)
    print("FINAL ASSISTANT RESPONSE")
    print("=" * 60)

    if result["success"]:

        print(result["answer"])

    else:

        print("Assistant failed:", result.get("error"))
