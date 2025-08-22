#!/usr/bin/env python3
"""
LoRA Fine-tuning Pipeline for Financial QA System
Implements LoRA fine-tuning with model merging and single safetensor saving
"""

import os
import json
import logging
import torch
import time
import shutil
import warnings
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Suppress specific warnings more comprehensively
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*can be ignored when running in bare mode.*")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Additional suppression for streamlit threading warnings
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

# Set environment variable to suppress streamlit warnings
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

# Transformers and PEFT
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    default_data_collator
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, PeftConfig
)
from datasets import Dataset, DatasetDict, concatenate_datasets
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalLMDataCollator:
    """Pad sequences and labels for Causal LM training with label masking.

    Ensures input_ids/attention_mask are padded to the longest length in batch and
    labels are padded with -100. This avoids shape mismatches on mixed-length examples.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels_list = [f["labels"] for f in features]
        enc_features = [
            {k: v for k, v in f.items() if k in ("input_ids", "attention_mask")}
            for f in features
        ]

        batch = self.tokenizer.pad(
            enc_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].size(1)
        padded_labels: List[List[int]] = []
        for lab in labels_list:
            cur = list(lab)
            if len(cur) < max_len:
                cur = cur + [-100] * (max_len - len(cur))
            else:
                cur = cur[:max_len]
            padded_labels.append(cur)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

@dataclass
class FineTuningConfig:
    """Configuration for LoRA fine-tuning"""
    # Base model
    base_model_path: str = "models/Llama-3.1-8B-Instruct"
    
    # Tuning method: 'lora' (PEFT) or 'adapter' (AdapterHub Pfeiffer)
    tuning_method: str = "lora"

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    # Adapter configuration (used when tuning_method == 'adapter')
    adapter_type: str = "pfeiffer"
    adapter_reduction_factor: int = 32  # Increased for more stability (less compression)
    adapter_non_linearity: str = "relu"
    
    # Training configuration
    learning_rate: float = 1e-5  # Community-informed conservative (lower bound of standard range)
    num_epochs: int = 3
    batch_size: int = 1  # Keep at 1 to avoid batching issues
    gradient_accumulation_steps: int = 4
    max_length: int = 1024  # Reduce to avoid memory issues
    
    # Advanced settings
    use_quantization: bool = False
    use_8bit_training: bool = False  # Enable 8-bit quantization for training (reduces VRAM)
    use_gradient_checkpointing: bool = False  # Disabled for LoRA compatibility
    warmup_steps: int = 100
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 50
    save_total_limit: int = 2  # Only keep last N checkpoints to save disk space
    max_grad_norm: float = 0.5  # More aggressive gradient clipping to prevent divergence
    weight_decay: float = 0.01  # L2 regularization for stability
    warmup_ratio: float = 0.1  # Add warmup period for gentler training start
    
    # Early stopping settings
    use_early_stopping: bool = True
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    
    # Output settings
    output_dir: str = "models/llama31-financial-qa-lora"
    merged_model_dir: str = "models/llama31-financial-qa-merged"
    
    # Data settings
    dataset_repetitions: int = 2  # repeat training data to strengthen learning on small datasets
    
    def __post_init__(self):
        if self.target_modules is None:
            # Phi-4 specific target modules
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingProgress:
    """Training progress tracking"""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    progress_percentage: float = 0.0
    estimated_time_remaining: str = "Unknown"
    status: str = "Not Started"

class DatasetHandler:
    """Handles dataset loading and preprocessing for financial Q&A"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_qa_dataset(self, dataset_path: str) -> Dataset:
        """Load Q&A dataset from JSON file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different dataset formats
            qa_pairs = []
            
            if isinstance(data, list):
                # Direct list of Q&A pairs
                qa_pairs = data
            elif isinstance(data, dict):
                # Check for various nested formats
                if 'financial_qa_pairs' in data:
                    # Our specific financial dataset format
                    qa_pairs = data['financial_qa_pairs']
                elif 'data' in data:
                    # General nested format
                    qa_pairs = data['data']
                elif 'questions' in data:
                    # Questions format
                    qa_pairs = data['questions']
                else:
                    # Assume the dict itself is a single Q&A pair
                    qa_pairs = [data]
            else:
                raise ValueError("Unsupported dataset format")
            
            # Convert to training format
            processed_data = []
            for item in qa_pairs:
                if isinstance(item, dict):
                    # Support multiple question/answer key formats
                    question = (item.get('instruction', '') or 
                              item.get('question', '') or 
                              item.get('Q', '') or
                              item.get('query', ''))
                    
                    answer = (item.get('output', '') or 
                             item.get('answer', '') or 
                             item.get('A', '') or
                             item.get('response', ''))
                    
                    # Handle input field if present (for instruction format)
                    input_text = item.get('input', '')
                    if input_text and input_text.strip():
                        question = f"{question}\n\nContext: {input_text}"
                else:
                    # Handle other formats
                    continue
                
                if question and answer:
                    processed_data.append({
                        'question': question,
                        'answer': answer,
                        'text': self._format_qa_pair(question, answer)
                    })
            
            logger.info(f"Loaded {len(processed_data)} Q&A pairs from {dataset_path}")
            return Dataset.from_list(processed_data)
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def load_from_docs_for_rag(self, docs_path: str = "data/docs_for_rag") -> Dataset:
        """Create dataset from docs_for_rag directory"""
        try:
            qa_pairs = []
            
            # Load financial_qa_rag.txt if it exists
            qa_file = os.path.join(docs_path, "financial_qa_rag.txt")
            if os.path.exists(qa_file):
                with open(qa_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse Q&A format
                lines = content.strip().split('\n')
                current_q = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Q: '):
                        current_q = line[3:]
                    elif line.startswith('A: ') and current_q:
                        current_a = line[3:]
                        qa_pairs.append({
                            'question': current_q,
                            'answer': current_a,
                            'text': self._format_qa_pair(current_q, current_a)
                        })
                        current_q = None
            
            logger.info(f"Created {len(qa_pairs)} Q&A pairs from docs_for_rag")
            return Dataset.from_list(qa_pairs) if qa_pairs else None
            
        except Exception as e:
            logger.error(f"Error creating dataset from docs_for_rag: {e}")
            return None
    
    def _format_qa_pair(self, question: str, answer: str) -> str:
        """Format Q&A pair for training (Llama 3.1 chat template)"""
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful AI assistant specialized in financial analysis. Answer questions accurately based on financial data.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"{answer}<|eot_id|>"
        )
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for training"""
        def tokenize_function(example):
            full_text = example['text']
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )

            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            labels = [-100] * len(input_ids)
            seq_len = len(input_ids)

            # Mask labels so loss is computed only on assistant answer tokens
            try:
                # Support both Phi-style and Llama 3.x chat templates
                llama_assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>"
                phi_assistant_hdr = "<|assistant|>"

                start_char_idx = -1
                end_char_idx = -1

                if llama_assistant_hdr in full_text:
                    start_char_idx = full_text.find(llama_assistant_hdr)
                    start_char_idx += len(llama_assistant_hdr)
                    if start_char_idx < len(full_text) and full_text[start_char_idx] == "\n":
                        start_char_idx += 1
                    # Try to stop at the end-of-turn token if present
                    eot_tag = "<|eot_id|>"
                    eot_pos = full_text.find(eot_tag, start_char_idx)
                    end_char_idx = eot_pos if eot_pos != -1 else len(full_text)
                elif phi_assistant_hdr in full_text:
                    start_char_idx = full_text.find(phi_assistant_hdr)
                    start_char_idx += len(phi_assistant_hdr)
                    if start_char_idx < len(full_text) and full_text[start_char_idx] == "\n":
                        start_char_idx += 1
                    end_char_idx = len(full_text)

                if start_char_idx != -1:
                    # More robust approach: find assistant response start by tokenizing components
                    # Split the text into parts before and after assistant header
                    prefix_text = full_text[:start_char_idx]
                    response_text = full_text[start_char_idx:end_char_idx] if end_char_idx != -1 else full_text[start_char_idx:]
                    
                    # Tokenize prefix to find where assistant response starts
                    prefix_tokens = self.tokenizer(
                        prefix_text,
                        truncation=True,
                        padding=False,
                        max_length=self.max_length,
                        return_tensors=None,
                        add_special_tokens=False  # Important: don't add extra tokens
                    )['input_ids']
                    
                    start_token = len(prefix_tokens)
                    
                    # For end token, tokenize the full prefix + response
                    if end_char_idx != -1 and end_char_idx <= len(full_text):
                        full_prefix_response = full_text[:end_char_idx]
                        full_tokens = self.tokenizer(
                            full_prefix_response,
                            truncation=True,
                            padding=False,
                            max_length=self.max_length,
                            return_tensors=None,
                            add_special_tokens=False
                        )['input_ids']
                        end_token = len(full_tokens)
                    else:
                        end_token = len(input_ids)

                    # Clamp to actual sequence length
                    start_token = max(0, min(start_token, len(input_ids)))
                    end_token = max(start_token, min(end_token, len(input_ids)))

                    # Set labels for the assistant response tokens
                    num_labeled_tokens = 0
                    for i in range(start_token, end_token):
                        labels[i] = input_ids[i]
                        num_labeled_tokens += 1
                    
                    # Debug: log first few examples
                    if hasattr(self, '_tokenize_debug_counter'):
                        self._tokenize_debug_counter += 1
                    else:
                        self._tokenize_debug_counter = 1
                    
                    if self._tokenize_debug_counter <= 5:
                        logger.info(f"Tokenize Debug {self._tokenize_debug_counter}:")
                        logger.info(f"  Total tokens: {len(input_ids)}")
                        logger.info(f"  Labeled tokens: {num_labeled_tokens} (from {start_token} to {end_token})")
                        logger.info(f"  Response text: '{response_text[:100]}...'")
                        
                        # Verify by decoding the labeled portion
                        labeled_tokens = [token_id for i, token_id in enumerate(input_ids) if labels[i] != -100]
                        if labeled_tokens:
                            decoded_labeled = self.tokenizer.decode(labeled_tokens, skip_special_tokens=True)
                            logger.info(f"  Decoded labeled: '{decoded_labeled[:100]}...'")
            
            except Exception as e:
                logger.warning(f"Error in label masking: {str(e)}")
                # Fallback: label the last 50% of tokens (rough heuristic)
                start_fallback = len(input_ids) // 2
                for i in range(start_fallback, len(input_ids)):
                    labels[i] = input_ids[i]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'length': seq_len,
            }
        
        # Process each example individually
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset, test_size: float = 0.2, val_size: float = 0.1) -> DatasetDict:
        """Stratified-ish split by length buckets to reduce distribution shift.

        We bucket by sequence length quantiles, then perform splits to keep similar
        length distributions across splits. This helps stabilize eval metrics.
        """
        try:
            # Create buckets
            lengths = dataset["length"]
            quantiles = np.quantile(lengths, [0.33, 0.66])
            def bucket(example):
                l = example["length"]
                if l <= quantiles[0]:
                    return 0
                elif l <= quantiles[1]:
                    return 1
                return 2
            with_bucket = dataset.add_column("_bucket", list(map(lambda x: bucket(x), dataset)))

            # Split within each bucket then concatenate
            def split_bucket(b):
                sub = with_bucket.filter(lambda e: e["_bucket"] == b)
                tv = sub.train_test_split(test_size=test_size, seed=42)
                val_ratio = val_size / (1 - test_size)
                tr_va = tv["train"].train_test_split(test_size=val_ratio, seed=42)
                return tr_va["train"], tr_va["test"], tv["test"]

            trains, vals, tests = [], [], []
            for b in [0, 1, 2]:
                tr_b, va_b, te_b = split_bucket(b)
                trains.append(tr_b)
                vals.append(va_b)
                tests.append(te_b)

            # Concatenate splits from buckets correctly (don't overwrite lists)
            trains = [ds for ds in trains if len(ds) > 0]
            vals = [ds for ds in vals if len(ds) > 0]
            tests = [ds for ds in tests if len(ds) > 0]

            train_final = concatenate_datasets(trains) if len(trains) > 1 else trains[0]
            val_final = concatenate_datasets(vals) if len(vals) > 1 else vals[0]
            test_final = concatenate_datasets(tests) if len(tests) > 1 else tests[0]

            # Clean helper column
            train_final = train_final.remove_columns(["_bucket"]) if "_bucket" in train_final.column_names else train_final
            val_final = val_final.remove_columns(["_bucket"]) if "_bucket" in val_final.column_names else val_final
            test_final = test_final.remove_columns(["_bucket"]) if "_bucket" in test_final.column_names else test_final

            return DatasetDict({
                'train': train_final,
                'validation': val_final,
                'test': test_final
            })
        except Exception:
            # Fallback to simple random split
            train_val_dataset = dataset.train_test_split(test_size=test_size, seed=42)
            val_test_size = val_size / (1 - test_size)
            train_dataset = train_val_dataset['train'].train_test_split(
                test_size=val_test_size, seed=42
            )
            return DatasetDict({
                'train': train_dataset['train'],
                'validation': train_dataset['test'],
                'test': train_val_dataset['test']
            })

class LoRATrainer:
    """LoRA fine-tuning trainer"""
    
    def __init__(self, config: FineTuningConfig, progress_callback: Optional[Callable] = None):
        self.config = config
        self.progress_callback = progress_callback
        self.training_progress = TrainingProgress()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None
        
        # Training state
        self.training_start_time = None
        self.is_training = False
        self.should_stop = False
    
    def setup_model_and_tokenizer(self):
        """Setup base model, tokenizer, and LoRA configuration"""
        try:
            self.training_progress.status = "Loading base model..."
            self._update_progress()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            
            # Set pad token to eos token for causal LM (required)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Ensure padding side is correct for causal LM
            self.tokenizer.padding_side = "right"
            # Keep the end of the sequence (assistant answer) when truncating
            try:
                self.tokenizer.truncation_side = "left"
            except Exception:
                pass
            
            # Load base model with optional 8-bit quantization
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "trust_remote_code": False,  # Avoid custom model code compatibility issues
                "low_cpu_mem_usage": True
            }
            
            # Add 8-bit quantization if enabled (QLoRA approach)
            if self.config.use_8bit_training:
                try:
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig
                    
                    # Use proper BitsAndBytesConfig for QLoRA
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("🔬 QLoRA 8-bit quantization enabled for training (VRAM optimization)")
                except ImportError:
                    logger.warning("⚠️ bitsandbytes not available. Install with: pip install bitsandbytes")
                    logger.info("Falling back to 16-bit training")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                **model_kwargs
            )
            
            # Note: Gradient checkpointing can interfere with adapter/LoRA training
            # Disable for parameter-efficient fine-tuning to avoid gradient issues

            # Check compatibility with 8-bit training for adapters
            if self.config.tuning_method.lower() == "adapter" and self.config.use_8bit_training:
                logger.warning("❌ Adapter training is not compatible with 8-bit quantization")
                logger.info("💡 QLoRA (8-bit + LoRA) is the recommended approach for quantized training")
                logger.info("🔄 Automatically switching to LoRA mode for 8-bit compatibility")
                self.config.tuning_method = "lora"
            
            if self.config.tuning_method.lower() == "adapter":
                # Adapter-based PEFT via AdapterHub (new 'adapters' package)
                try:
                    from adapters import AdapterConfig
                    from adapters import init as adapters_init
                except Exception as e:
                    raise RuntimeError(
                        "Adapters library is required for adapter tuning. Install it with: pip install -U adapters"
                    ) from e

                # Initialize adapter hooks for the current model
                try:
                    adapters_init(self.model)
                except Exception:
                    pass

                adapter_config = AdapterConfig.load(
                    self.config.adapter_type,
                    reduction_factor=self.config.adapter_reduction_factor,
                    non_linearity=self.config.adapter_non_linearity,
                )
                self.model.add_adapter("financial_qa", config=adapter_config)
                self.model.train_adapter("financial_qa")
                self.model.set_active_adapters("financial_qa")
                self.peft_model = self.model
                
                # Log adapter-specific info for debugging
                logger.info(f"Adapter Configuration:")
                logger.info(f"  Type: {self.config.adapter_type}")
                logger.info(f"  Reduction Factor: {self.config.adapter_reduction_factor}")
                logger.info(f"  Non-linearity: {self.config.adapter_non_linearity}")
                
                # Print trainable parameters for adapters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                logger.info(f"  Total parameters: {total_params:,}")
                logger.info(f"  Trainable parameters: {trainable_params:,}")
                logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
            else:
                # Default to LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules,
                    bias="none"
                )
                self.peft_model = get_peft_model(self.model, lora_config)
                self.peft_model.print_trainable_parameters()
                self.peft_model.train()
                for param in self.peft_model.parameters():
                    if param.requires_grad:
                        param.requires_grad_(True)
            
            logger.info("Model and tokenizer setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            self.training_progress.status = f"Error: {e}"
            self._update_progress()
            return False
    
    def prepare_dataset(self, dataset_path: str = None) -> bool:
        """Prepare dataset for training"""
        try:
            self.training_progress.status = "Preparing dataset..."
            self._update_progress()
            
            dataset_handler = DatasetHandler(self.tokenizer, self.config.max_length)
            
            # Load dataset
            if dataset_path and os.path.exists(dataset_path):
                dataset = dataset_handler.load_qa_dataset(dataset_path)
            else:
                # Try to load from docs_for_rag
                dataset = dataset_handler.load_from_docs_for_rag()
            
            if dataset is None or len(dataset) == 0:
                raise ValueError("No dataset found or dataset is empty")
            
            # Tokenize dataset
            tokenized_dataset = dataset_handler.tokenize_dataset(dataset)
            
            # Split dataset
            self.dataset_dict = dataset_handler.split_dataset(tokenized_dataset)
            
            # Optionally repeat the training split to increase effective data size
            try:
                reps = max(1, int(self.config.dataset_repetitions))
            except Exception:
                reps = 1
            if reps > 1:
                from datasets import concatenate_datasets
                repeats = [self.dataset_dict['train']]
                for _ in range(reps - 1):
                    repeats.append(self.dataset_dict['train'])
                self.dataset_dict['train'] = concatenate_datasets(repeats).shuffle(seed=42)
            
            logger.info(f"Dataset prepared: {len(self.dataset_dict['train'])} train, "
                       f"{len(self.dataset_dict['validation'])} val, "
                       f"{len(self.dataset_dict['test'])} test samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            self.training_progress.status = f"Dataset error: {e}"
            self._update_progress()
            return False
    
    def start_training(self) -> bool:
        """Start LoRA fine-tuning"""
        try:
            # Comprehensive warning suppression for training
            warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
            warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
            warnings.filterwarnings("ignore", message=".*can be ignored when running in bare mode.*")
            
            # Suppress streamlit warnings at logging level
            logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
            
            self.training_progress.status = "Starting training..."
            self.is_training = True
            self.training_start_time = time.time()
            self._update_progress()
            
            # Training arguments (prefer latest API: eval_strategy; fallback to evaluation_strategy if needed)
            use_fp16 = torch.cuda.is_available()
            use_bf16 = False
            try:
                if torch.cuda.is_available():
                    major, _ = torch.cuda.get_device_capability()
                    use_bf16 = major >= 8  # Ampere or newer
            except Exception:
                use_bf16 = False

            # Build version-safe TrainingArguments
            ta_params = inspect.signature(TrainingArguments.__init__).parameters
            strategy_key = 'eval_strategy' if 'eval_strategy' in ta_params else (
                'evaluation_strategy' if 'evaluation_strategy' in ta_params else None
            )

            kwargs = dict(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type="cosine",  # Use cosine decay instead of linear
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_strategy="epoch",
                save_total_limit=self.config.save_total_limit,  # Only keep last N checkpoints to save disk space
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                dataloader_pin_memory=True,
                remove_unused_columns=True,
                push_to_hub=False,
                report_to=[],
                dataloader_num_workers=0,
                fp16=(use_fp16 and not use_bf16),
                bf16=use_bf16,
                tf32=True if torch.cuda.is_available() else None,
                optim="adamw_torch",
                max_grad_norm=self.config.max_grad_norm,  # Gradient clipping
                weight_decay=self.config.weight_decay,  # L2 regularization
            )
            if strategy_key:
                kwargs[strategy_key] = "epoch"
            # For accurate token-level metrics, use logits aligned with labels
            if 'predict_with_generate' in ta_params:
                kwargs['predict_with_generate'] = False
            if 'generation_max_length' in ta_params:
                kwargs['generation_max_length'] = 128
            if 'generation_num_beams' in ta_params:
                kwargs['generation_num_beams'] = 1

            training_args = TrainingArguments(**kwargs)
            
            # Use custom collator to pad inputs and masked labels safely on HF 4.55
            data_collator = CausalLMDataCollator(self.tokenizer)
            
            # Create a stable progress callback using the official TrainerCallback
            progress_handler = CustomProgressCallback(
                progress_callback=self._on_training_step
            )

            # Metric: Jaccard similarity and token-level accuracy
            def _compute_metrics(eval_pred):
                try:
                    preds = eval_pred.predictions
                    labels = eval_pred.label_ids
                    
                    # predictions may be logits or token ids depending on HF version
                    if isinstance(preds, tuple):
                        preds = preds[0]

                    # Helper: trim sequence at EOS if present
                    def _trim_eos(seq_ids):
                        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
                        if eos_id is None:
                            return seq_ids
                        try:
                            idx = seq_ids.index(eos_id)
                            return seq_ids[:idx]
                        except ValueError:
                            return seq_ids

                    pred_texts = []
                    gold_texts = []
                    correct_tokens = 0
                    total_tokens = 0
                    
                    # Debug: Print shapes and sample data
                    if hasattr(self, '_debug_counter'):
                        self._debug_counter += 1
                    else:
                        self._debug_counter = 1
                    
                    if self._debug_counter <= 3:  # Only print first few times
                        logger.info(f"DEBUG - Eval batch {self._debug_counter}:")
                        logger.info(f"  Predictions shape: {preds.shape}")
                        logger.info(f"  Labels shape: {labels.shape}")
                        logger.info(f"  Predictions ndim: {preds.ndim}")

                    if preds.ndim == 3:
                        # Logits aligned with labels (teacher-forced evaluation)
                        preds_ids = preds.argmax(-1)
                        
                        for i, (p_row, l_row) in enumerate(zip(preds_ids, labels)):
                            l_list = l_row.tolist()
                            # Find positions where we should compute loss (labels != -100)
                            keep_idxs = [j for j, t in enumerate(l_list) if t != -100]
                            
                            if not keep_idxs:
                                pred_texts.append("")
                                gold_texts.append("")
                                if self._debug_counter <= 3:
                                    logger.info(f"  Sample {i}: No valid labels (all -100)")
                                continue
                            
                            # Fix: Handle causal LM shift - predictions[i] predicts labels[i+1]
                            # Only compare positions where both pred[i] and label[i+1] are valid
                            valid_pairs = []
                            for j in range(len(keep_idxs) - 1):  # -1 because we look ahead
                                curr_idx = keep_idxs[j]
                                next_idx = keep_idxs[j + 1]
                                
                                # Check if next position is consecutive (for proper causal alignment)
                                if next_idx == curr_idx + 1:
                                    pred_token = int(p_row[curr_idx])  # prediction at position i
                                    gold_token = int(l_list[next_idx])  # gold token at position i+1
                                    valid_pairs.append((pred_token, gold_token))
                            
                            # If no valid pairs, fall back to direct comparison (edge case)
                            if not valid_pairs and keep_idxs:
                                p_kept = [int(p_row[j]) for j in keep_idxs]
                                g_kept = [int(l_list[j]) for j in keep_idxs]
                                valid_pairs = list(zip(p_kept, g_kept))
                            
                            # Extract tokens for debugging and text generation
                            p_kept = [int(p_row[j]) for j in keep_idxs]
                            g_kept = [int(l_list[j]) for j in keep_idxs]
                            
                            # Debug first sample
                            if self._debug_counter <= 3 and i == 0:
                                logger.info(f"  Sample {i}: {len(keep_idxs)} supervised tokens, {len(valid_pairs)} valid pairs")
                                logger.info(f"  Gold tokens (first 10): {g_kept[:10]}")
                                logger.info(f"  Pred tokens (first 10): {p_kept[:10]}")
                                if valid_pairs:
                                    logger.info(f"  First 5 pred->gold pairs: {valid_pairs[:5]}")
                                gold_text = self.tokenizer.decode(g_kept, skip_special_tokens=True)
                                pred_text = self.tokenizer.decode(p_kept, skip_special_tokens=True)
                                logger.info(f"  Gold text: '{gold_text[:100]}...'")
                                logger.info(f"  Pred text: '{pred_text[:100]}...'")
                            
                            # Calculate token-level accuracy using valid pairs
                            total_tokens += len(valid_pairs)
                            matches = sum(1 for pk, gk in valid_pairs if pk == gk)
                            correct_tokens += matches
                            
                            # Store decoded texts for Jaccard calculation
                            pred_texts.append(self.tokenizer.decode(p_kept, skip_special_tokens=True))
                            gold_texts.append(self.tokenizer.decode(g_kept, skip_special_tokens=True))
                    
                    else:
                        # Generated token ids per sample (generation-based evaluation)
                        # This branch should rarely be used in teacher-forced training
                        logger.warning("Using generation-based evaluation - this may not be optimal for training")
                        
                        preds_ids = preds
                        for i, (p_row, l_row) in enumerate(zip(preds_ids, labels)):
                            # Extract gold answer tokens (labels without -100)
                            g_ids = [int(t) for t in l_row.tolist() if t != -100]
                            # Process predicted token IDs
                            p_ids = list(map(int, list(p_row)))
                            p_ids = _trim_eos(p_ids)

                            # Calculate token-level accuracy up to minimum length
                            m = min(len(p_ids), len(g_ids))
                            if m > 0:
                                total_tokens += m
                                correct_tokens += sum(1 for j in range(m) if p_ids[j] == g_ids[j])

                            pred_texts.append(self.tokenizer.decode(p_ids, skip_special_tokens=True))
                            gold_texts.append(self.tokenizer.decode(g_ids, skip_special_tokens=True))

                    # Calculate Jaccard similarity
                    def jaccard(a, b):
                        sa = set((a or "").lower().split())
                        sb = set((b or "").lower().split())
                        if not sa and not sb:
                            return 1.0
                        if not sa or not sb:
                            return 0.0
                        inter = len(sa & sb)
                        union = len(sa | sb)
                        return inter / union if union else 0.0

                    sims = [jaccard(p, g) for p, g in zip(pred_texts, gold_texts)]
                    avg_jaccard = float(np.mean(sims)) if sims else 0.0
                    token_acc = (float(correct_tokens) / float(total_tokens)) if total_tokens > 0 else 0.0
                    
                    # Debug output
                    if self._debug_counter <= 3:
                        logger.info(f"  Total tokens: {total_tokens}")
                        logger.info(f"  Correct tokens: {correct_tokens}")
                        logger.info(f"  Token accuracy: {token_acc}")
                        logger.info(f"  Jaccard similarity: {avg_jaccard}")
                    
                    # Log if token accuracy is still problematic (but be less alarmist now that we fixed the bug)
                    if token_acc == 0.0 and total_tokens > 0:
                        logger.info(f"📊 Token accuracy: 0% (Total valid pairs: {total_tokens}, Batch size: {len(pred_texts)})")
                        logger.info("ℹ️ Note: Token accuracy uses strict causal LM alignment. Check Jaccard similarity for semantic quality.")
                        if len(pred_texts) > 0 and len(gold_texts) > 0:
                            logger.info(f"Sample pred text: '{pred_texts[0][:100]}...'")
                            logger.info(f"Sample gold text: '{gold_texts[0][:100]}...'")
                    elif token_acc > 0.0:
                        logger.info(f"🎯 Improved token accuracy: {token_acc:.1%} (Total valid pairs: {total_tokens})")                    
                    
                    # Log overall training health with more detailed diagnostics
                    if hasattr(self, '_prev_token_acc'):
                        if token_acc < self._prev_token_acc * 0.5:  # Dropped by more than 50%
                            logger.warning(f"🔥 Large token accuracy drop: {self._prev_token_acc:.4f} → {token_acc:.4f}")
                            logger.warning("   This suggests training instability - consider stopping and adjusting hyperparameters")
                    
                    # Track accuracy trend for early intervention
                    if not hasattr(self, '_token_acc_history'):
                        self._token_acc_history = []
                    self._token_acc_history.append(token_acc)
                    
                    # If we have 3+ evaluations and all recent ones are getting worse, warn
                    if len(self._token_acc_history) >= 3:
                        recent_trend = self._token_acc_history[-3:]
                        if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
                            logger.warning(f"⚠️ Token accuracy declining for 3 consecutive evaluations: {recent_trend}")
                            logger.warning("   Strong indication of training divergence - consider stopping training")
                    
                    self._prev_token_acc = token_acc
                    
                    return {
                        "jaccard": avg_jaccard,
                        "token_accuracy": token_acc
                    }
                    
                except Exception as e:
                    logger.error(f"Error in compute_metrics: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"jaccard": 0.0, "token_accuracy": 0.0}

            # Build callbacks list (conditionally include early stopping)
            callbacks = [progress_handler]
            if self.config.use_early_stopping:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=int(self.config.early_stopping_patience),
                        early_stopping_threshold=float(self.config.early_stopping_threshold)
                    )
                )

            # Use the standard Trainer and pass our callback to it
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=self.dataset_dict['train'],
                eval_dataset=self.dataset_dict['validation'],
                data_collator=data_collator,
                callbacks=callbacks,
                compute_metrics=_compute_metrics
            )
            
            # Calculate total steps
            total_samples = len(self.dataset_dict['train'])
            steps_per_epoch = max(1, total_samples // max(1, (self.config.batch_size * self.config.gradient_accumulation_steps)))
            self.training_progress.total_steps = steps_per_epoch * self.config.num_epochs
            
            # Start training
            logger.info("Starting LoRA fine-tuning...")
            self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            self.training_progress.status = "Training completed!"
            self.is_training = False
            self._update_progress()
            
            logger.info("LoRA fine-tuning completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.training_progress.status = f"Training error: {e}"
            self.is_training = False
            self._update_progress()
            return False
    
    def merge_and_save_model(self) -> bool:
        """Merge LoRA weights with base model and save as single safetensor"""
        try:
            self.training_progress.status = "Merging LoRA weights..."
            self._update_progress()
            
            # Load the fine-tuned LoRA model for merging (use 16-bit for merging regardless of training quantization)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False,
                low_cpu_mem_usage=True
                # Note: Always use 16-bit for merging as merged model should be full precision
            )
            
            # Load and merge LoRA weights
            merged_model = PeftModel.from_pretrained(base_model, self.config.output_dir)
            merged_model = merged_model.merge_and_unload()
            
            # Save merged model
            os.makedirs(self.config.merged_model_dir, exist_ok=True)
            merged_model.save_pretrained(
                self.config.merged_model_dir,
                safe_serialization=True  # Save as safetensors
            )
            
            # Save tokenizer (use the base tokenizer to match merged model vocab)
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            tokenizer.save_pretrained(self.config.merged_model_dir)

            # Copy architecture files
            self._copy_architecture_files()
            
            # Save model info
            model_info = {
                "base_model": self.config.base_model_path,
                "fine_tuned_on": "financial_qa_dataset",
                "lora_config": {
                    "r": self.config.lora_r,
                    "alpha": self.config.lora_alpha,
                    "dropout": self.config.lora_dropout,
                    "target_modules": self.config.target_modules
                },
                "training_config": {
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size
                },
                "merged_at": datetime.now().isoformat()
            }
            
            with open(os.path.join(self.config.merged_model_dir, "model_info.json"), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.training_progress.status = "Model merged and saved successfully!"
            self._update_progress()
            
            logger.info(f"Merged model saved to {self.config.merged_model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging model: {e}")
            self.training_progress.status = f"Merge error: {e}"
            self._update_progress()
            return False
    
    def stop_training(self):
        """Stop training"""
        self.should_stop = True
        if self.trainer:
            self.trainer.control.should_training_stop = True
    
    def _on_training_step(self, logs: Dict[str, Any]):
        """Called on each training step"""
        if self.should_stop:
            return
        
        # Update progress
        if 'epoch' in logs:
            self.training_progress.epoch = int(logs['epoch'])
        if 'step' in logs:
            self.training_progress.step = logs['step']
        if 'loss' in logs:
            self.training_progress.loss = logs['loss']
        if 'learning_rate' in logs:
            self.training_progress.learning_rate = logs['learning_rate']
        
        # Calculate progress percentage
        if self.training_progress.total_steps > 0:
            self.training_progress.progress_percentage = (
                self.training_progress.step / self.training_progress.total_steps * 100
            )
        
        # Estimate time remaining
        if self.training_start_time and self.training_progress.step > 0:
            elapsed_time = time.time() - self.training_start_time
            steps_per_second = self.training_progress.step / elapsed_time
            remaining_steps = self.training_progress.total_steps - self.training_progress.step
            if steps_per_second > 0:
                remaining_seconds = remaining_steps / steps_per_second
                self.training_progress.estimated_time_remaining = self._format_time(remaining_seconds)
        
        self.training_progress.status = "Training in progress..."
        self._update_progress()
    
    def _update_progress(self):
        """Update progress via callback"""
        if self.progress_callback:
            # Convert TrainingProgress to dict for callback compatibility
            progress_dict = {
                'epoch': self.training_progress.epoch,
                'step': self.training_progress.step,
                'total_steps': self.training_progress.total_steps,
                'loss': self.training_progress.loss,
                'learning_rate': self.training_progress.learning_rate,
                'progress_percentage': self.training_progress.progress_percentage,
                'estimated_time_remaining': self.training_progress.estimated_time_remaining,
                'status': self.training_progress.status
            }
            self.progress_callback(progress_dict)
    
    def _copy_architecture_files(self):
        """Copy any custom architecture files from base to merged model directory (if they exist)"""
        try:
            base_path = self.config.base_model_path
            merged_path = self.config.merged_model_dir
            
            # Look for any custom modeling/configuration Python files in the base model directory
            # This is only needed for models with custom architectures (like some Phi models)
            # Standard models like Llama don't need this
            if not os.path.exists(base_path):
                logger.debug("Base model path not found, skipping architecture file copy")
                return
                
            custom_files_found = False
            for file_name in os.listdir(base_path):
                # Only copy custom architecture files (modeling_*.py, configuration_*.py)
                if (file_name.startswith(("modeling_", "configuration_")) and 
                    file_name.endswith(".py")):
                    
                    source_file = os.path.join(base_path, file_name)
                    dest_file = os.path.join(merged_path, file_name)
                    
                    if os.path.isfile(source_file):
                        shutil.copy2(source_file, dest_file)
                        logger.info(f"Copied custom architecture file: {file_name}")
                        custom_files_found = True
            
            if not custom_files_found:
                logger.debug("No custom architecture files found - using standard HuggingFace model")
                    
        except Exception as e:
            logger.error(f"Error copying architecture files: {e}")

    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

class CustomProgressCallback(TrainerCallback):
    """
    A robust, version-agnostic callback to handle training progress updates.
    This hooks into the 'on_log' event, which is the correct way to get
    progress data from the Trainer.
    """
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback

    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        if self.progress_callback and logs is not None:
            # The 'logs' dict from the trainer contains loss, learning_rate, etc.
            # We add the step and epoch from the trainer's state for completeness.
            progress_data = logs.copy()
            progress_data['step'] = state.global_step
            progress_data['epoch'] = state.epoch
            self.progress_callback(progress_data)

    def on_evaluate(self, args: TrainingArguments, state, control, metrics=None, **kwargs):
        # Surface evaluation metrics (e.g., eval_loss) to the UI callback
        if self.progress_callback and metrics is not None:
            eval_data = metrics.copy()
            eval_data['step'] = state.global_step
            eval_data['epoch'] = state.epoch
            eval_data['status'] = 'Evaluation'
            self.progress_callback(eval_data)

# Utility functions for easy usage

def create_fine_tuning_config(**kwargs) -> FineTuningConfig:
    """Create fine-tuning configuration with custom parameters.
    Defaults output directories based on tuning method to avoid misleading names.
    """
    tuning_method = str(kwargs.get("tuning_method", "lora")).lower()
    # Set sensible defaults if caller didn't provide custom dirs
    if "output_dir" not in kwargs or not kwargs.get("output_dir"):
        kwargs["output_dir"] = (
            "models/llama31-financial-qa-adapter" if tuning_method == "adapter" else "models/llama31-financial-qa-lora"
        )
    if "merged_model_dir" not in kwargs or not kwargs.get("merged_model_dir"):
        # Note: adapter flow generally doesn't merge into base; keep a distinct name anyway
        kwargs["merged_model_dir"] = (
            "models/llama31-financial-qa-adapter-merged" if tuning_method == "adapter" else "models/llama31-financial-qa-merged"
        )
    return FineTuningConfig(**kwargs)

def start_lora_fine_tuning(config: FineTuningConfig, 
                          dataset_path: str = None,
                          progress_callback: Callable = None) -> bool:
    """Start LoRA fine-tuning with given configuration"""
    trainer = LoRATrainer(config, progress_callback)
    
    # Setup model
    if not trainer.setup_model_and_tokenizer():
        return False
    
    # Prepare dataset
    if not trainer.prepare_dataset(dataset_path):
        return False
    
    # Start training
    if not trainer.start_training():
        return False
    
    # Merge and save model
    if not trainer.merge_and_save_model():
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = FineTuningConfig(
        base_model_path="models/Llama-3.1-8B-Instruct",
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=1
    )
    
    # Progress callback
    def progress_callback(progress: TrainingProgress):
        print(f"Step {progress.step}/{progress.total_steps} - "
              f"Loss: {progress.loss:.4f} - "
              f"Progress: {progress.progress_percentage:.1f}% - "
              f"Status: {progress.status}")
    
    # Start fine-tuning
    success = start_lora_fine_tuning(
        config=config,
        dataset_path="data/dataset/financial_qa_finetune.json",
        progress_callback=progress_callback
    )
    
    if success:
        print("Fine-tuning completed successfully!")
    else:
        print("Fine-tuning failed!")
