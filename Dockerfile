FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py app.py ./

# Train model DURING build → savedmodel.pth inside image
RUN python train.py

EXPOSE 5000

CMD [\""python\"", \""app.py\""]