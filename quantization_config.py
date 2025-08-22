#!/usr/bin/env python3
"""
Modern Quantization Configuration for Fine-tuned Models

This module provides standardized quantization configurations using the 
modern BitsAndBytesConfig approach instead of deprecated parameters.
"""

import torch
from transformers import BitsAndBytesConfig

def get_8bit_config():
    """
    Get optimized 8-bit quantization configuration
    
    Returns:
        BitsAndBytesConfig: 8-bit quantization configuration
    """
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=False
    )

def get_4bit_config():
    """
    Get optimized 4-bit quantization configuration (QLoRA compatible)
    
    Returns:
        BitsAndBytesConfig: 4-bit quantization configuration
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def get_quantization_config(bit_mode: str = None):
    """
    Get quantization configuration based on mode
    
    Args:
        bit_mode: "8bit", "4bit", or None for no quantization
        
    Returns:
        BitsAndBytesConfig or None: Quantization configuration
    """
    if bit_mode == "8bit":
        return get_8bit_config()
    elif bit_mode == "4bit":
        return get_4bit_config()
    else:
        return None

# Example usage:
if __name__ == "__main__":
    print("=== Modern Quantization Configurations ===")
    
    print("\n8-bit Config:")
    config_8bit = get_8bit_config()
    print(f"  load_in_8bit: {config_8bit.load_in_8bit}")
    print(f"  llm_int8_threshold: {config_8bit.llm_int8_threshold}")
    
    print("\n4-bit Config:")
    config_4bit = get_4bit_config()
    print(f"  load_in_4bit: {config_4bit.load_in_4bit}")
    print(f"  bnb_4bit_quant_type: {config_4bit.bnb_4bit_quant_type}")
    
    print("\nUsage in model loading:")
    print("""
# OLD (deprecated)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True  # ❌ Deprecated, shows warnings
)

# NEW (recommended)
from quantization_config import get_8bit_config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=get_8bit_config()  # ✅ Modern approach
)
    """)
