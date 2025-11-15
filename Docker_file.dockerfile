# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY train.py .
COPY app.py .

# Train model inside container
RUN python train.py

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]