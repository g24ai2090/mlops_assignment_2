FROM python:3.12-slim

WORKDIR /app

# Install only scikit-learn + joblib + flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY train.py app.py ./

# TRAIN DURING BUILD → savedmodel.pth inside image
RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]