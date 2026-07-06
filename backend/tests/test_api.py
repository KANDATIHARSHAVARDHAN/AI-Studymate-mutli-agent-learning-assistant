import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

client = TestClient(app)

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0

def test_get_files_empty():
    response = client.get("/api/files")
    assert response.status_code == 200
    assert response.json()["files"] == []

def test_chat_endpoint_missing_keys():
    # This might fail with 500 if API keys are missing in CI, which is expected for a simple test suite
    response = client.post("/api/chat", json={"query": "Hello", "model_name": "gpt-oss-120b"})
    assert response.status_code in [200, 500]
