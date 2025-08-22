#!/usr/bin/env python3
"""
Baseline Evaluation Script for Pre-trained Llama 3.1 8B
Evaluates the model before fine-tuning for assignment comparison
"""

import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    """Evaluate pre-trained model performance for assignment baseline"""
    
    def __init__(self, model_path: str = "models/Llama-3.1-8B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load pre-trained model for baseline evaluation"""
        logger.info(f"Loading baseline model: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ Baseline model loaded successfully")
        
    def evaluate_question(self, question: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate a single question and return metrics"""
        start_time = time.time()
        
        # Create chat template
        messages = [
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip()
        
        inference_time = time.time() - start_time
        
        # Calculate basic similarity (can be enhanced)
        similarity = self._calculate_similarity(response, expected_answer)
        
        return {
            "question": question,
            "generated_answer": response,
            "expected_answer": expected_answer,
            "similarity_score": similarity,
            "inference_time_seconds": inference_time,
            "confidence_score": 0.5  # Placeholder - can be enhanced with actual confidence
        }
    
    def _calculate_similarity(self, generated: str, expected: str) -> float:
        """Calculate similarity between generated and expected answers"""
        # Simple word overlap similarity (can be enhanced with semantic similarity)
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        
        if not exp_words:
            return 0.0
            
        overlap = len(gen_words.intersection(exp_words))
        return overlap / len(exp_words)
    
    def evaluate_dataset(self, test_questions: List[Dict[str, str]], num_questions: int = 10) -> Dict[str, Any]:
        """Evaluate on a subset of test questions"""
        logger.info(f"Evaluating baseline on {num_questions} questions...")
        
        results = []
        total_similarity = 0.0
        total_time = 0.0
        
        for i, qa_pair in enumerate(test_questions[:num_questions]):
            logger.info(f"Evaluating question {i+1}/{num_questions}")
            
            result = self.evaluate_question(
                qa_pair["instruction"], 
                qa_pair["output"]
            )
            results.append(result)
            total_similarity += result["similarity_score"]
            total_time += result["inference_time_seconds"]
        
        # Calculate summary metrics
        avg_similarity = total_similarity / len(results)
        avg_inference_time = total_time / len(results)
        
        summary = {
            "model_name": "Llama-3.1-8B-Instruct (Pre-trained)",
            "num_questions_evaluated": len(results),
            "average_similarity_score": avg_similarity,
            "average_inference_time": avg_inference_time,
            "total_evaluation_time": total_time,
            "detailed_results": results
        }
        
        logger.info(f"✅ Baseline evaluation complete!")
        logger.info(f"📊 Average similarity: {avg_similarity:.3f}")
        logger.info(f"⏱️ Average inference time: {avg_inference_time:.3f}s")
        
        return summary

def main():
    """Run baseline evaluation for assignment"""
    
    # Load test dataset
    with open("data/dataset/financial_qa_finetune.json", "r") as f:
        dataset = json.load(f)
    
    # Take last 50 as test set (first part used for training)
    test_questions = dataset[-50:]
    
    # Initialize evaluator
    evaluator = BaselineEvaluator()
    evaluator.load_model()
    
    # Run baseline evaluation
    results = evaluator.evaluate_dataset(test_questions, num_questions=15)
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    
    with open("results/baseline_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary for assignment
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY FOR ASSIGNMENT")
    print("="*60)
    print(f"Model: {results['model_name']}")
    print(f"Questions Evaluated: {results['num_questions_evaluated']}")
    print(f"Average Accuracy: {results['average_similarity_score']:.1%}")
    print(f"Average Inference Time: {results['average_inference_time']:.3f} seconds")
    print(f"Compute Setup: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    print("\nSAMPLE RESULTS:")
    for i, result in enumerate(results['detailed_results'][:3]):
        print(f"\n- Question {i+1}: {result['question']}")
        print(f"  Generated: {result['generated_answer'][:100]}...")
        print(f"  Similarity: {result['similarity_score']:.2%}")
        print(f"  Time: {result['inference_time_seconds']:.3f}s")

if __name__ == "__main__":
    main()
