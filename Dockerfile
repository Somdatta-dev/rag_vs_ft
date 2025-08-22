# Multi-stage Dockerfile for RAG vs Fine-Tuning Financial QA System
# Optimized for Coolify deployment with GPU support

# Stage 1: Base image with CUDA support for ML workloads
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Stage 2: Python dependencies installation
FROM base AS dependencies

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application files
FROM dependencies AS application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/dataset data/docs_for_rag data/test \
             results models notebooks ui screenshots database/init

# Set proper permissions
RUN chmod +x setup.py && \
    chmod -R 755 /app

# Stage 4: Production image
FROM application AS production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Environment variables for production
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting RAG vs Fine-Tuning Financial QA System"\n\
echo "GPU Available: $(python -c \"import torch; print(torch.cuda.is_available())\")" \n\
echo "CUDA Devices: $(python -c \"import torch; print(torch.cuda.device_count())\")" \n\
\n\
# Check if models exist, if not download them\n\
if [ ! -d "models/Llama-3.1-8B-Instruct" ] || [ ! -d "models/mxbai-embed-large-v1" ]; then\n\
    echo "⬇️ Models not found, running setup..."\n\
    python setup.py\nfi\n\
\n\
# Start Streamlit\n\
echo "🌐 Starting Streamlit on port 8501"\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Labels for better maintenance
LABEL maintainer="Financial QA System"
LABEL description="RAG vs Fine-Tuning Financial QA System with GPU support"
LABEL version="1.0"
LABEL gpu.required="true"
