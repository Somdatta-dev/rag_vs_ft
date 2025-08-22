# Coolify Demo Deployment Guide - CPU Only

## 🎭 Demo Mode Deployment (No GPU Required)

This guide shows how to deploy the Financial QA System UI demo on any VPS without GPU requirements.

### Prerequisites
- Coolify instance (no GPU needed)
- VPS with 4GB+ RAM and 2+ CPU cores
- Docker support

### 1. Repository Setup

```bash
# Add demo files to your repository
git add Dockerfile.demo docker-compose.demo.yml requirements-demo.txt demo-mode.py
git commit -m "Add CPU-only demo deployment configuration"
git push origin main
```

### 2. Coolify Configuration

1. **Create New Application** in Coolify dashboard
2. **Select Git Repository** and connect your repo
3. **Application Settings**:
   ```
   Application Type: Docker
   Branch: main
   Dockerfile: Dockerfile.demo
   ```

### 3. Environment Variables

In Coolify, add these environment variables:

```env
# Demo Mode Settings
DEMO_MODE=true
USE_GPU=false

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
PYTHONUNBUFFERED=1
```

### 4. Resource Requirements (Minimal)

Configure in Coolify:
```yaml
Deploy Configuration:
  Memory: 4GB
  CPU: 2 cores
  Storage: 10GB
  GPU: Not required
```

### 5. Port Configuration

- **Internal Port**: 8501
- **External Port**: 80 or 443 (handled by Coolify)
- **Custom Domain**: Configure in Coolify dashboard

### 6. Build Configuration

```yaml
Build Settings:
  Build Command: docker build -f Dockerfile.demo -t financial-qa-demo .
  Start Command: /app/start-demo.sh
  Build Context: ./
```

### 7. What Works in Demo Mode

✅ **Fully Functional UI**:
- All tabs and navigation
- Complete Streamlit interface
- Form inputs and buttons
- Charts and visualizations

✅ **Mock Responses**:
- RAG pipeline returns realistic sample answers
- Fine-tuned model provides demo responses
- Comparison functionality with mock metrics
- Baseline evaluation with sample data

✅ **Interactive Features**:
- Document processing interface (mock)
- Fine-tuning interface (simulation)
- Model comparison (mock results)
- Performance metrics (demo data)

❌ **What's Disabled**:
- Actual model inference
- Real document processing
- GPU-based operations
- Large model downloads

### 8. Demo Responses

The demo mode provides realistic sample responses:

**RAG Demo Response**:
```
"Based on the financial documents, Accenture's total revenue 
for fiscal year 2024 was $64.9 billion, representing a 1% 
increase in U.S. dollars..."

Confidence: 92%
Sources: Accenture FY2024 Q4 Earnings Report
Response Time: 2.34s (simulated)
```

**Fine-tuned Demo Response**:
```
"Accenture's net revenue for fiscal 2024 was $64.9 billion, 
an increase of 1% in U.S. dollars and 4% in local currency..."

Confidence: 89%
Model: Llama-3.1-8B Fine-tuned (LoRA r=256)
Response Time: 1.45s (simulated)
```

### 9. Deployment Commands

**Using Docker Compose locally**:
```bash
# Test demo locally
docker-compose -f docker-compose.demo.yml up --build

# Access at http://localhost:8501
```

**Direct Docker build**:
```bash
# Build demo image
docker build -f Dockerfile.demo -t financial-qa-demo .

# Run demo container
docker run -p 8501:8501 -e DEMO_MODE=true financial-qa-demo
```

### 10. Cost-Effective VPS Options

**Recommended VPS Specs**:
- **RAM**: 4GB
- **CPU**: 2 cores
- **Storage**: 10GB
- **Bandwidth**: Unlimited
- **Cost**: $10-20/month

**Popular VPS Providers**:
- DigitalOcean ($12/month)
- Linode ($12/month)
- Vultr ($12/month)
- Hetzner ($8/month)

### 11. Performance Expectations

**Build Time**: 3-5 minutes
**Container Size**: ~2GB (vs 15GB+ for full version)
**Startup Time**: 10-20 seconds
**Memory Usage**: 1-2GB RAM
**Response Time**: 1-3 seconds (simulated)

### 12. Monitoring

```bash
# Check container status
docker ps

# View logs
docker logs <container-name>

# Monitor resources
docker stats
```

### 13. Customization

You can customize demo responses in `demo-mode.py`:

```python
# Add your own demo responses
demo_responses = [
    {
        "answer": "Your custom demo response here...",
        "confidence": 0.95,
        "sources": ["Custom Source"],
        "response_time": 2.1
    }
]
```

### 14. Access Your Demo

After deployment:
```
Demo URL: https://your-domain.com
Health Check: https://your-domain.com/_stcore/health
```

## 🎯 Perfect for:

- **Client demonstrations**
- **UI/UX showcasing**
- **Proof of concept presentations**
- **Portfolio demonstrations**
- **Academic presentations**
- **Budget-friendly hosting**

The demo provides a fully interactive experience showcasing all UI features without the computational overhead of actual AI models! 🚀
