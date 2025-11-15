FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/model.joblib src/dictvectorizer.joblib src/train.py src/predict.py src/serve.py ./

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

# NEXT:
# To build your Docker image

# Open VS Code terminal:

# docker build -t pet-adoption-api .

# docker build -t pet-adoption-model .

# docker run -p 8000:8000 pet-adoption-model



# Run the container:

# docker run -p 8000:8000 pet-adoption-api


# API opens at:

# http://localhost:8000/docs
