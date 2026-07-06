import os
import sys
import pytest
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Ensure the backend directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import classify_query

load_dotenv()

@pytest.fixture(scope="module")
def llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0,
        max_tokens=100
    )

test_queries = [
    # Direct/indirect MCQ
    ("I want to practice some multiple choice questions on YOLO", "mcq_generator"),
    ("test my knowledge on yolo", "mcq_generator"),
    ("give me a quiz on yolo", "mcq_generator"),
    
    # Direct/indirect Exam Prep
    ("prepare a question paper on yolo", "exam_prep_agent"),
    ("what are some likely questions I will get on my yolo exam", "exam_prep_agent"),
    ("design a study guide for yolo test", "exam_prep_agent"),
    
    # Direct/indirect Notes
    ("make some study notes about YOLOv8", "notes_maker"),
    ("give me a cheat sheet for YOLO concept", "notes_maker"),
    
    # Direct/indirect Explanation
    ("explain how yolo does anchor boxes", "concept_explainer"),
    ("what is the definition of intersection over union", "concept_explainer"),
    
    # Direct/indirect Search
    ("what is the latest release date of yolov11", "search_agent"),
    ("search the web for yolo accuracy improvements", "search_agent"),
    
    # Direct/indirect Summarize
    ("can you condense this document for me", "summarizer"),
    ("give me a brief summary of yolo architecture", "summarizer"),
    
    # Chat
    ("hello, can you help me today?", "chat_agent"),
    ("thanks a lot, bye", "chat_agent")
]

@pytest.mark.parametrize("query, expected", test_queries)
def test_classify_query(llm, query, expected):
    actual = classify_query(query, llm)
    assert actual == expected, f"Query '{query}' failed. Expected '{expected}', got '{actual}'"
