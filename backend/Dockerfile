# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
