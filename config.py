# Configuration for RAG vs Fine-Tuning System

# Model paths
PHI4_MODEL_PATH = "models/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_PATH = "models/mxbai-embed-large-v1"

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
DATASET_PATH = "data/dataset"
DOCS_FOR_RAG_PATH = "data/docs_for_rag"

# RAG settings
CHUNK_SIZES = [100, 400]
TOP_K_RETRIEVAL = 5
EMBEDDING_DIMENSION = 512

# LoRA Fine-tuning settings (optimized for 100 QA pairs)
LORA_LEARNING_RATE = 1e-4
LORA_BATCH_SIZE = 1
LORA_NUM_EPOCHS = 3
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
MAX_LENGTH = 16000
USE_QUANTIZATION = True

# Standard Fine-tuning settings
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
NUM_EPOCHS = 3

# RAG Generation settings
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.3
CONTEXT_LENGTH = 16000

# UI settings
STREAMLIT_PORT = 8501
