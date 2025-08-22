#!/usr/bin/env python3
"""
Demo Mode Handler for Financial QA System
Provides mock responses and UI functionality without actual model inference
"""

import json
import time
import random
from typing import Dict, Any, List
import os

class DemoRAGPipeline:
    """Mock RAG pipeline for demo purposes"""
    
    def __init__(self):
        self.demo_responses = [
            {
                "answer": "Based on the financial documents, Accenture's total revenue for fiscal year 2024 was $64.9 billion, representing a 1% increase in U.S. dollars and 4% increase in local currency compared to fiscal 2023.",
                "confidence": 0.92,
                "sources": ["Accenture FY2024 Q4 Earnings Report", "Annual Financial Statement"],
                "retrieval_time": 0.85,
                "response_time": 2.34
            },
            {
                "answer": "According to the retrieved documents, Accenture's operating margin for Q4 FY2024 improved to 15.8%, up from 15.1% in the same quarter of the previous year, driven by strong operational efficiency and cost management.",
                "confidence": 0.88,
                "sources": ["Q4 2024 Financial Results", "Operating Performance Analysis"],
                "retrieval_time": 0.92,
                "response_time": 2.67
            },
            {
                "answer": "The documents indicate that Accenture's Technology services growth was 3% in local currency for fiscal 2024, with particularly strong performance in cloud and data analytics solutions.",
                "confidence": 0.85,
                "sources": ["Technology Services Report", "Cloud Revenue Analysis"],
                "retrieval_time": 1.12,
                "response_time": 3.01
            }
        ]
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Mock query method that returns demo responses"""
        # Simulate processing time
        time.sleep(random.uniform(2.0, 3.5))
        
        # Return random demo response
        response = random.choice(self.demo_responses)
        response["method"] = "RAG (Hybrid Retrieval)"
        response["question"] = question
        
        return response

class DemoFinetunedPipeline:
    """Mock fine-tuned pipeline for demo purposes"""
    
    def __init__(self):
        self.demo_responses = [
            {
                "answer": "Accenture's net revenue for fiscal 2024 was $64.9 billion, an increase of 1% in U.S. dollars and 4% in local currency compared to fiscal 2023. This growth was driven by strong performance across all geographic markets.",
                "confidence": 0.89,
                "response_time": 1.45,
                "model_info": "Llama-3.1-8B Fine-tuned (LoRA r=256)"
            },
            {
                "answer": "For fiscal year 2024, Accenture reported diluted earnings per share of $12.73 on a GAAP basis and $13.40 on an adjusted basis, representing strong profitability and shareholder value creation.",
                "confidence": 0.91,
                "response_time": 1.23,
                "model_info": "Llama-3.1-8B Fine-tuned (LoRA r=256)"
            },
            {
                "answer": "Accenture's Strategy & Consulting revenue grew 6% in local currency for fiscal 2024, demonstrating strong demand for strategic advisory services and digital transformation consulting.",
                "confidence": 0.86,
                "response_time": 1.67,
                "model_info": "Llama-3.1-8B Fine-tuned (LoRA r=256)"
            }
        ]
    
    def query(self, question: str, max_length: int = 128, temperature: float = 0.1) -> Dict[str, Any]:
        """Mock query method that returns demo responses"""
        # Simulate processing time (slightly faster than RAG)
        time.sleep(random.uniform(1.0, 2.0))
        
        # Return random demo response
        response = random.choice(self.demo_responses)
        response["method"] = "Fine-tuned Model"
        response["question"] = question
        
        return response

def create_demo_data():
    """Create sample data files for demo mode"""
    
    # Sample Q&A data
    demo_qa_data = {
        "financial_qa_pairs": [
            {
                "instruction": "What was Accenture's total revenue for fiscal year 2024?",
                "output": "Accenture's total revenue for fiscal year 2024 was $64.9 billion, representing a 1% increase in U.S. dollars and 4% increase in local currency compared to fiscal 2023."
            },
            {
                "instruction": "What were Accenture's GAAP earnings per share for fiscal 2024?",
                "output": "Accenture's GAAP diluted earnings per share for fiscal 2024 were $12.73, compared to $11.05 in fiscal 2023."
            },
            {
                "instruction": "How did Accenture's Technology services perform in fiscal 2024?",
                "output": "Technology services revenue grew 3% in local currency for fiscal 2024, driven by strong demand for cloud and data analytics solutions."
            }
        ]
    }
    
    # Test questions
    demo_test_data = [
        {
            "instruction": "What was the operating margin for Q4 FY2024?",
            "output": "The operating margin for Q4 FY2024 was 15.8%, up from 15.1% in the same quarter of the previous year."
        },
        {
            "instruction": "What is Accenture's revenue outlook for fiscal 2025?",
            "output": "Accenture projects revenue growth of 2-5% in local currency for fiscal 2025, with continued strength in Technology and Strategy & Consulting."
        }
    ]
    
    # Create directories
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/docs_for_rag", exist_ok=True)
    
    # Write demo data files
    with open("data/dataset/financial_qa_finetune.json", "w") as f:
        json.dump(demo_qa_data, f, indent=2)
    
    with open("data/test/comprehensive_qa.json", "w") as f:
        json.dump(demo_test_data, f, indent=2)
    
    # Create sample RAG document
    sample_doc = """
    ACCENTURE FISCAL YEAR 2024 FINANCIAL HIGHLIGHTS
    
    Total Revenue: $64.9 billion (1% growth in USD, 4% in local currency)
    Operating Margin: 15.8% in Q4, improved from 15.1% prior year
    GAAP EPS: $12.73 for fiscal 2024
    Technology Services: 3% growth in local currency
    Strategy & Consulting: 6% growth in local currency
    
    This is sample data for demonstration purposes.
    """
    
    with open("data/docs_for_rag/financial_qa_rag.txt", "w") as f:
        f.write(sample_doc)
    
    print("✅ Demo data files created successfully")

def is_demo_mode() -> bool:
    """Check if application is running in demo mode"""
    return os.getenv("DEMO_MODE", "false").lower() == "true"

if __name__ == "__main__":
    create_demo_data()
    print("Demo mode setup complete!")
