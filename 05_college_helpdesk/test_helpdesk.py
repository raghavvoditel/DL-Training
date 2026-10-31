# Automatic tests for the helpdesk.  Run with:  pytest test_helpdesk.py -v

import pytest

from helpdesk import Helpdesk


@pytest.fixture(scope="module")
def desk():
    return Helpdesk()


@pytest.mark.parametrize("question, expected_source, expected_words", [
    ("What time does the library open?", "library.txt", "8:30 am"),
    ("How many books can I borrow?", "library.txt", "4 books"),
    ("What is the fine for late return of a book?", "library.txt", "5 rupees"),
    ("How much attendance is required to write the exam?", "exams.txt", "75 percent"),
    ("What is the passing mark?", "exams.txt", "40 out of 100"),
    ("When do hostel gates close?", "hostel.txt", "10:00 pm"),
    ("What are the mess timings for dinner?", "hostel.txt", "7:30 pm"),
    ("What is the tuition fee?", "fees.txt", "1,20,000"),
    ("How can I pay fees online?", "fees.txt", "UPI"),
])
def test_finds_right_answer(desk, question, expected_source, expected_words):
    result = desk.answer(question)
    assert result["source"] == expected_source
    assert expected_words in result["answer"]


def test_unknown_question(desk):
    result = desk.answer("Who won the football world cup?")
    assert result["source"] is None
    assert "Sorry" in result["answer"]
