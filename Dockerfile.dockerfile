# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY train.py test.py app.py ./

# --- TRAIN & QUANTIZE MODEL DURING BUILD ---
RUN python train.py

# Expose port
EXPOSE 5000

# Run Flask
CMD ["python", "app.py"]