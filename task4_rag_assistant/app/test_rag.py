
import pytest

from task4_rag_assistant.app.rag import ask_question


# ==========================================
# 6 ANSWERABLE QUESTIONS
# ==========================================

ANSWERABLE_QUESTIONS = [
    "What is SupportDesk Pro?",
    "What is the purpose of SupportDesk Pro?",
    "How does SupportDesk Pro help businesses?",
    "What features are available in the product?",
    "How can users manage their customer support requests?",
    "What should users do when they experience a login problem?"
]

# ==========================================
# 4 UNANSWERABLE QUESTIONS
# ==========================================

UNANSWERABLE_QUESTIONS = [
    "Who is the CEO of Microsoft?",
    "What is the weather today in Chennai?",
    "What is the current price of an iPhone?",
    "Who won the FIFA World Cup in 2022?"
]


# ==========================================
# TEST 1: ANSWERABLE QUESTIONS
# ==========================================

@pytest.mark.parametrize("question", ANSWERABLE_QUESTIONS)
def test_answerable_questions(question):

    result = ask_question(question)

    print("\nANSWERABLE QUESTION:", question)
    print("ANSWER:", result["answer"])
    print("SOURCES:", result["sources"])

    # Answer must not be "I don't know"
    assert (
        result["answer"].strip().lower().rstrip(".")
        != "i don't know"
    )

    # Sources must be returned
    assert len(result["sources"]) > 0


# ==========================================
# TEST 2: UNANSWERABLE QUESTIONS
# ==========================================

@pytest.mark.parametrize("question", UNANSWERABLE_QUESTIONS)
def test_unanswerable_questions(question):

    result = ask_question(question)

    print("\nUNANSWERABLE QUESTION:", question)
    print("ANSWER:", result["answer"])

    # Answer must be "I don't know"
    assert (
        result["answer"].strip().lower().rstrip(".")
        == "i don't know"
    )


# ==========================================
# TEST SUMMARY
# ==========================================

def test_question_count():

    assert len(ANSWERABLE_QUESTIONS) == 6
    assert len(UNANSWERABLE_QUESTIONS) == 4

    print("\nTotal answerable questions: 6")
    print("Total unanswerable questions: 4")
    print("Total test cases: 10")
