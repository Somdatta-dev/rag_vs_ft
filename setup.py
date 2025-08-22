#!/usr/bin/env python3
"""
Centralized Setup Script for RAG vs Fine-Tuning Financial QA System
Creates directory structure, downloads models, and configures LoRA fine-tuning
"""

import os
import sys
import subprocess
import platform
import time
import socket
from pathlib import Path
import urllib.request
import shutil

def run_command(command, shell=True):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        print(f"Success: {command}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return None

def create_directory_structure():
    """Create the required directory structure"""
    print("Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/dataset",
        "data/docs_for_rag",
        "models",
        "src",
        "notebooks",
        "results",
        "ui",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_python_version():
    """Check if Python 3.12 is available"""
    print("Checking Python version...")
    
    # Check current Python version
    current_version = sys.version_info
    print(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    
    if current_version.major == 3 and current_version.minor >= 12:
        print("Python 3.12+ is available")
        return sys.executable
    
    # Try to find Python 3.12
    python_commands = ["python3.12", "python3", "python"]
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if "Python 3.12" in result.stdout:
                print(f"Found Python 3.12: {cmd}")
                return cmd
        except FileNotFoundError:
            continue
    
    print("Warning: Python 3.12 not found. Using current Python version.")
    return sys.executable

def is_valid_venv(venv_path):
    """Check if virtual environment is valid"""
    if not venv_path.exists():
        return False
    
    # Check for essential files
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return pyvenv_cfg.exists() and python_exe.exists()

def create_virtual_environment(python_cmd):
    """Create virtual environment"""
    print("Creating virtual environment...")
    
    venv_path = Path("venv")
    
    # Check if virtual environment exists and is valid
    if venv_path.exists():
        if is_valid_venv(venv_path):
            print("Valid virtual environment already exists. Skipping creation...")
            return True
        else:
            print("Existing virtual environment is corrupted. Removing and recreating...")
            try:
                shutil.rmtree(venv_path)
            except PermissionError:
                print("Cannot remove corrupted virtual environment due to permission error.")
                print("Please manually delete the 'venv' folder and run setup again.")
                return False
    
    # Create virtual environment
    result = run_command(f"{python_cmd} -m venv venv")
    if not result:
        print("Failed to create virtual environment")
        return False
    
    print("Virtual environment created")
    return True

def get_venv_python():
    """Get the path to Python in virtual environment"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python.exe"
    else:
        return "venv/bin/python"

def get_venv_pip():
    """Get the path to pip in virtual environment"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip.exe"
    else:
        return "venv/bin/pip"

def install_pytorch():
    """Install PyTorch with CUDA 12.8 support"""
    print("Installing PyTorch with CUDA 12.8...")
    
    python_cmd = get_venv_python()
    
    # Check if virtual environment python exists
    if not Path(python_cmd).exists():
        print("Virtual environment Python not found. Using system Python...")
        python_cmd = sys.executable
    
    pytorch_cmd = f"{python_cmd} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    
    result = run_command(pytorch_cmd)
    if result:
        print("PyTorch with CUDA 12.8 installed successfully")
        return True
    else:
        print("Failed to install PyTorch")
        return False

def install_lora_dependencies():
    """Install LoRA fine-tuning dependencies"""
    print("Installing LoRA fine-tuning dependencies...")
    
    python_cmd = get_venv_python()
    
    # Check if virtual environment python exists
    if not Path(python_cmd).exists():
        print("Virtual environment Python not found. Using system Python...")
        python_cmd = sys.executable
    
    # LoRA specific packages
    lora_packages = [
        "peft",
        "bitsandbytes",
        "accelerate>=0.20.0"
    ]
    
    success_count = 0
    for package in lora_packages:
        print(f"Installing {package}...")
        result = run_command(f"{python_cmd} -m pip install {package}")
        if result:
            success_count += 1
            print(f"[✓] Successfully installed {package}")
        else:
            print(f"[X] Failed to install {package}")
    
    print(f"LoRA dependencies: {success_count}/{len(lora_packages)} installed successfully")
    return success_count == len(lora_packages)

def install_requirements():
    """Install requirements from requirements.txt"""
    print("Installing base requirements...")
    
    python_cmd = get_venv_python()
    
    # Check if virtual environment python exists
    if not Path(python_cmd).exists():
        print("Virtual environment Python not found. Using system Python...")
        python_cmd = sys.executable
    
    # Upgrade pip first using python -m pip
    run_command(f"{python_cmd} -m pip install --upgrade pip")
    
    # Install requirements
    result = run_command(f"{python_cmd} -m pip install -r requirements.txt")
    if result:
        print("Base requirements installed successfully")
        return True
    else:
        print("Failed to install base requirements")
        return False

def download_models():
    """Download the specified models using huggingface-hub"""
    print("Downloading models...")
    
    python_cmd = get_venv_python()
    
    # Check if virtual environment python exists
    if not Path(python_cmd).exists():
        print("Virtual environment Python not found. Using system Python...")
        python_cmd = sys.executable
    
    # Create model download script
    download_script = '''
import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(repo_id, local_dir):
    """Download a model from Hugging Face Hub"""
    try:
        # Check if model already exists
        model_path = Path(local_dir)
        if model_path.exists() and any(model_path.iterdir()):
            print(f"Model {repo_id} already exists at {local_dir}. Skipping download...")
            return True
            
        print(f"Downloading {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN")
        )
        print(f"Downloaded {repo_id} to {local_dir}")
        return True
    except Exception as e:
        print(f"Failed to download {repo_id}: {str(e)}")
        return False

# Download models
models = [
    ("meta-llama/Llama-3.1-8B-Instruct", "models/Llama-3.1-8B-Instruct"),
    ("mixedbread-ai/mxbai-embed-large-v1", "models/mxbai-embed-large-v1")
]

success_count = 0
for repo_id, local_dir in models:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    if download_model(repo_id, local_dir):
        success_count += 1

print(f"Downloaded {success_count}/{len(models)} models successfully")
'''
    
    # Write and execute download script
    with open("download_models.py", "w", encoding='utf-8') as f:
        f.write(download_script)
    
    result = run_command(f"{python_cmd} download_models.py")
    
    # Clean up
    if os.path.exists("download_models.py"):
        os.remove("download_models.py")
    
    if result:
        print("Models downloaded successfully")
        return True
    else:
        print("Failed to download models")
        return False

def check_docker():
    """Check if Docker is installed and running"""
    print("Checking Docker installation...")
    
    # Check if docker command exists
    result = run_command("docker --version")
    if not result:
        print("Docker is not installed. Please install Docker first.")
        return False
    
    # Check if Docker is running
    result = run_command("docker info")
    if not result:
        print("Docker is not running. Please start Docker first.")
        return False
    
    print("Docker is installed and running")
    return True

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("Checking Docker Compose...")
    
    # Try docker compose (newer version)
    result = run_command("docker compose version")
    if result:
        print("Docker Compose (v2) is available")
        return "docker compose"
    
    # Try docker-compose (older version)
    result = run_command("docker-compose --version")
    if result:
        print("Docker Compose (v1) is available")
        return "docker-compose"
    
    print("Docker Compose is not available")
    return None

def check_port_availability(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def setup_database():
    """Setup PostgreSQL database with Docker Compose"""
    print("Setting up PostgreSQL database with pgvector...")
    
    # Check Docker prerequisites
    if not check_docker():
        print("[!] Docker not available. Skipping database setup.")
        print("   You can set up the database later by running: python setup_database.py")
        return False
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        print("[!] Docker Compose not available. Skipping database setup.")
        return False
    
    # Create database directory
    Path("database/init").mkdir(parents=True, exist_ok=True)
    
    # Stop existing services first
    print("Stopping any existing database services...")
    run_command(f"{compose_cmd} down", shell=True)
    time.sleep(3)
    
    # Start database services
    print("Starting PostgreSQL database services...")
    result = run_command(f"{compose_cmd} up -d")
    if not result:
        print("Failed to start database services")
        return False
    
    # Wait for database to be ready
    print("Waiting for database to be ready...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Try to connect to the database
            python_cmd = get_venv_python()
            if not Path(python_cmd).exists():
                python_cmd = sys.executable
            
            # Install psycopg2 if not available
            try:
                import psycopg2
            except ImportError:
                print("Installing psycopg2-binary...")
                run_command(f"{python_cmd} -m pip install psycopg2-binary")
                import psycopg2
            
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                database="rag_financial_qa",
                user="rag_user",
                password="rag_password"
            )
            conn.close()
            print("[✓] Database is ready!")
            return True
            
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt}/{max_attempts}: Database not ready yet...")
            time.sleep(2)
    
    print("[X] Database failed to start within expected time")
    return False

def test_cuda_availability():
    """Test CUDA availability for LoRA fine-tuning"""
    print("Testing CUDA availability for LoRA fine-tuning...")
    
    try:
        python_cmd = get_venv_python()
        if not Path(python_cmd).exists():
            python_cmd = sys.executable
        
        # Test CUDA with a simple script
        cuda_test_script = '''
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA_AVAILABLE:True")
        print(f"DEVICE_COUNT:{device_count}")
        print(f"DEVICE_NAME:{device_name}")
        print(f"MEMORY_GB:{memory_gb:.1f}")
    else:
        print("CUDA_AVAILABLE:False")
except Exception as e:
    print(f"CUDA_ERROR:{str(e)}")
'''
        
        with open("test_cuda.py", "w") as f:
            f.write(cuda_test_script)
        
        result = run_command(f"{python_cmd} test_cuda.py")
        
        # Clean up
        if os.path.exists("test_cuda.py"):
            os.remove("test_cuda.py")
        
        if result and "CUDA_AVAILABLE:True" in result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith("DEVICE_NAME:"):
                    device_name = line.split(":", 1)[1]
                    print(f"[✓] CUDA available: {device_name}")
                elif line.startswith("MEMORY_GB:"):
                    memory = float(line.split(":", 1)[1])
                    print(f"GPU Memory: {memory:.1f} GB")
                    if memory < 8:
                        print("[!] Low GPU memory - will use 4-bit quantization")
            return True
        else:
            print("[!] CUDA not available - LoRA will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"[X] Error testing CUDA: {e}")
        return False

def create_sample_files():
    """Create sample configuration and starter files"""
    print("Creating sample files...")
    
    # Create config file
    config_content = '''# Configuration for RAG vs Fine-Tuning System

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
'''
    
    with open("config.py", "w", encoding='utf-8') as f:
        f.write(config_content)
    
    # Note: app.py and gui.py are created separately with comprehensive layout
    print("Note: app.py and gui.py files should be created separately for the full GUI layout")
    
    # Create comprehensive README
    readme_content = '''# RAG vs Fine-Tuning Financial QA System

A comprehensive system comparing RAG (Retrieval-Augmented Generation) and LoRA Fine-tuning approaches for financial question answering.

## Quick Start

1. **Run centralized setup:**
   ```bash
   python setup.py
   ```

2. **Activate virtual environment:**
   ```bash
   # Windows
   venv\\Scripts\\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

## Directory Structure

```
├── data/
│   ├── raw/                    # Raw financial documents
│   ├── processed/              # Processed text data
│   ├── dataset/                # Q&A datasets for fine-tuning
│   └── docs_for_rag/          # Documents for RAG system
├── models/
│   ├── Llama-3.1-8B-Instruct/ # Base language model (Meta Llama 3.1 8B Instruct)
│   ├── mxbai-embed-large-v1/  # Embedding model
│   └── lora_financial_qa/     # LoRA fine-tuned model
├── src/                       # Source code modules
└── results/                   # Evaluation results
```

## For Your 100 QA Pairs

### LoRA Fine-tuning (Recommended)
- **Perfect for 100 QA pairs** - parameter efficient
- **Fast training** - 8-20 minutes
- **Low memory** - 8-12GB GPU with quantization
- **High quality** - maintains model performance

### Usage:
1. Place your dataset at: `data/dataset/financial_qa_finetune.json`
2. Use the "Fine-Tune" tab in the Streamlit interface
3. Select "LoRA (Recommended)"
4. Adjust parameters if needed
5. Start training!

## System Components

### Models Used
- **Meta Llama-3.1-8B-Instruct (8B)** - Text generation & instruction following
- **MxBai Large v1** - Embeddings (1024 dimensions)
- **PyTorch with CUDA 12.8** - GPU acceleration

### LoRA Configuration
- **Rank (r)**: 16 - Good balance of efficiency/performance
- **Alpha**: 32 - Scaling parameter
- **Learning Rate**: 1e-4 - Optimized for larger model
- **Quantization**: 4-bit - Reduces memory by 75%

## Performance Expectations

### LoRA Fine-tuning
- **Training Time**: 8-20 minutes (100 QA pairs)
- **Memory Usage**: 8-12GB GPU (with quantization)
- **Model Size**: Base model + 64MB adapter
- **Accuracy**: High on financial domain questions

### RAG System
- **Response Time**: 2-5 seconds
- **Memory Usage**: 4-6GB GPU
- **Accuracy**: Depends on document quality
- **Scalability**: Easy to add new documents

## Testing Your Setup

```bash
# Test model configuration
python test_model_config.py

# Run comprehensive tests
python -m pytest tests/
```

## Documentation

- `LORA_FINETUNING_README.md` - Complete LoRA guide
- `RAG_GUI_USAGE.md` - RAG system usage
- `DOCUMENT_PROCESSING_README.md` - Document processing

## Troubleshooting

### Common Issues:
1. **CUDA Out of Memory**: Reduce batch size, enable quantization
2. **Model Download Fails**: Check internet connection

### Get Help:
- Check the specific README files for detailed guides
- Use the test scripts to diagnose issues
- Review the Streamlit interface for real-time feedback

## Success Indicators

- All models downloaded successfully  
- LoRA dependencies installed  
- CUDA available (optional but recommended)  
- Streamlit app launches without errors  

Happy fine-tuning!
'''
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Sample files created")

def show_setup_summary(lora_success, cuda_available):
    """Show setup summary and next steps"""
    print("\n" + "=" * 70)
    print("SETUP COMPLETED!")
    print("=" * 70)
    
    print("\nSetup Summary:")
    print("[✓] Directory structure created")
    print("[✓] Virtual environment configured")
    print("[✓] PyTorch with CUDA 12.8 installed")
    print("[✓] Base requirements installed")
    print("[✓] Models downloaded")
    print("[✓] Configuration files created")
    
    if lora_success:
        print("[✓] LoRA fine-tuning dependencies installed")
    else:
        print("[!] LoRA dependencies partially installed")
    
    if cuda_available:
        print("[✓] CUDA available for GPU acceleration")
    else:
        print("[!] CUDA not available (CPU-only mode)")
    
    print("\nNext Steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run the application:")
    print("   streamlit run app.py")
    
    print("\n3. Use the application:")
    print("   • 'Process Documents' tab - Upload financial documents")
    print("   • 'Fine-Tune' tab - Run LoRA fine-tuning on your 100 QA pairs")
    print("   • 'Inference' tab - Test RAG vs Fine-tuned models")
    
    print("\nDocumentation:")
    print("   • LORA_FINETUNING_README.md - LoRA fine-tuning guide")
    print("   • RAG_GUI_USAGE.md - RAG system usage")
    print("   • DOCUMENT_PROCESSING_README.md - Document processing")
    
    print("\nFor your 100 QA pairs:")
    print("   • Place dataset at: data/dataset/financial_qa_finetune.json")
    print("   • Use 'Fine-Tune' tab with LoRA (Recommended)")
    print("   • Expected training time: 8-20 minutes")
    print("   • Memory usage: ~8-12GB GPU (with quantization)")
    
    print("=" * 70)

def main():
    """Main centralized setup function"""
    print("=" * 70)
    print("RAG vs Fine-Tuning Financial QA System - Complete Setup")
    print("=" * 70)
    
    setup_results = {
        'lora': False,
        'cuda': False
    }
    
    try:
        # Step 1: Basic setup
        print("\nStep 1: Basic Environment Setup")
        print("-" * 40)
        
        python_cmd = check_python_version()
        create_directory_structure()
        
        if not create_virtual_environment(python_cmd):
            print("Setup failed at virtual environment creation")
            return
        
        # Step 2: Install PyTorch
        print("\nStep 2: PyTorch Installation")
        print("-" * 40)
        
        if not install_pytorch():
            print("Setup failed at PyTorch installation")
            return
        
        # Step 3: Install base requirements
        print("\nStep 3: Base Requirements")
        print("-" * 40)
        
        if not install_requirements():
            print("Setup failed at base requirements installation")
            return
        
        # Step 4: Install LoRA dependencies
        print("\nStep 4: LoRA Fine-tuning Dependencies")
        print("-" * 40)
        
        setup_results['lora'] = install_lora_dependencies()
        
        # Step 5: Test CUDA
        print("\nStep 5: CUDA Testing")
        print("-" * 40)
        
        setup_results['cuda'] = test_cuda_availability()
        
        # Step 6: Model Download
        print("\nStep 6: Model Download")
        print("-" * 40)
        
        if not download_models():
            print("Setup failed at model download")
            return
        
        # Step 7: Create configuration files
        print("\nStep 7: Configuration Files")
        print("-" * 40)
        
        create_sample_files()
        
        # Show summary
        show_setup_summary(
            setup_results['lora'], 
            setup_results['cuda']
        )
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        print("You can resume setup by running: python setup.py")
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        print("Please check the error above and try again")

if __name__ == "__main__":
    main()