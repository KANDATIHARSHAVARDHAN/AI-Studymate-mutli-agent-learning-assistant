"""
Ragas Evaluation Module
"""
import sys
import langchain_core
# Ragas 0.4.x relies on langchain_core.pydantic_v1 which was removed in langchain-core 0.3.x. This hack fixes it.
if not hasattr(langchain_core, "pydantic_v1"):
    try:
        from pydantic import v1 as pydantic_v1
    except ImportError:
        import pydantic as pydantic_v1
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
    langchain_core.pydantic_v1 = pydantic_v1

import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def evaluate_interaction(query: str, response: str, contexts: list):
    """
    Evaluates a single query-response pair using Ragas metrics (faithfulness, answer_relevancy).
    """
    if not contexts:
        return {"faithfulness": 0.0, "answer_relevance": 0.0}

    data_sample = {
        "question": [query],
        "answer": [response],
        "contexts": [[str(ctx) for ctx in contexts]],
    }
    
    dataset = Dataset.from_dict(data_sample)
    
    try:
        llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.environ.get("GROQ_API_KEY"), max_tokens=2000)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.environ.get("GEMINI_API_KEY"))
        
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        f_metric = Faithfulness(llm=ragas_llm)
        ar_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)

        score = evaluate(
            dataset, 
            metrics=[f_metric, ar_metric], 
            raise_exceptions=False
        )
        
        df = score.to_pandas()
        return {
            "faithfulness": float(df["faithfulness"].iloc[0]) if not pd.isna(df["faithfulness"].iloc[0]) else 0.0,
            "answer_relevance": float(df["answer_relevancy"].iloc[0]) if not pd.isna(df["answer_relevancy"].iloc[0]) else 0.0,
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"faithfulness": 0.0, "answer_relevance": 0.0}
