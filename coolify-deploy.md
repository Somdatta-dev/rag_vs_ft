# Coolify Deployment Guide for RAG vs Fine-Tuning Financial QA System

## 🚀 Quick Deployment with Coolify

### Prerequisites
- Coolify instance with GPU support
- Docker with NVIDIA Container Toolkit
- At least 32GB RAM and 24GB GPU VRAM (recommended)

### 1. Repository Setup

```bash
# Clone or push your repository to Git
git add Dockerfile docker-compose.yml .dockerignore env.example
git commit -m "Add Docker configuration for Coolify deployment"
git push origin main
```

### 2. Coolify Configuration

1. **Create New Application** in Coolify dashboard
2. **Select Git Repository** and connect your repo
3. **Application Settings**:
   ```
   Application Type: Docker Compose
   Branch: main
   Build Pack: Docker
   ```

### 3. Environment Variables

In Coolify, add these environment variables:

```env
# Required
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1

# Optional (customize as needed)
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USE_GPU=true
```

### 4. Resource Requirements

Configure in Coolify:
```yaml
Deploy Configuration:
  Memory: 32GB
  CPU: 8 cores
  GPU: 1x NVIDIA (24GB+ VRAM)
  Storage: 100GB SSD
```

### 5. Port Configuration

- **Internal Port**: 8501
- **External Port**: 80 or 443 (Coolify handles this)
- **Custom Domain**: Configure in Coolify dashboard

### 6. Deployment Steps

1. **Build Settings**:
   ```
   Build Command: docker build -t financial-qa-system .
   Start Command: /app/start.sh
   ```

2. **Volume Mounts** (for persistent models):
   ```
   /app/models -> persistent volume
   /app/data -> persistent volume  
   /app/results -> persistent volume
   ```

3. **Health Check**:
   ```
   Health Check URL: /_stcore/health
   Health Check Interval: 30s
   ```

### 7. SSL & Domain

Coolify automatically handles:
- SSL certificate generation
- Domain routing
- Load balancing

### 8. Monitoring

Access logs in Coolify:
```bash
# Application logs
docker logs <container-id>

# GPU monitoring
nvidia-smi

# Memory usage
docker stats
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **Memory Issues**:
   - Increase container memory limits
   - Enable model quantization in app
   - Use smaller models for testing

3. **Model Download Timeout**:
   - Pre-download models to persistent volume
   - Increase build timeout in Coolify

4. **Port Access Issues**:
   - Verify port 8501 is exposed
   - Check Coolify proxy configuration

### Performance Optimization

1. **Use GPU for inference**:
   ```python
   # In app, verify GPU usage
   import torch
   print(f"GPU Available: {torch.cuda.is_available()}")
   ```

2. **Model Caching**:
   - Mount persistent volumes for models
   - Use Docker layer caching

3. **Resource Monitoring**:
   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi
   
   # Monitor container resources
   docker stats financial-qa-system
   ```

## 📊 Expected Performance

With proper GPU setup:
- **Build Time**: 10-15 minutes (first build)
- **Startup Time**: 2-3 minutes (model loading)
- **RAG Response**: 2-4 seconds
- **Fine-tuned Response**: 1-2 seconds
- **Memory Usage**: 28-30GB RAM, 20-22GB VRAM

## 🔒 Security Considerations

1. **Environment Variables**: Use Coolify's secret management
2. **Network**: Configure firewall rules
3. **Updates**: Regular container updates via Coolify
4. **Backup**: Backup persistent volumes containing models

## 🌐 Access Your Application

After successful deployment:
```
URL: https://your-domain.com
Health Check: https://your-domain.com/_stcore/health
```

The application will be accessible via the domain configured in Coolify!
