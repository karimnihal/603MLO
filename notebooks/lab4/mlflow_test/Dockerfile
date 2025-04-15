FROM python:3.10-slim

WORKDIR /mlflow

# Install mlflow
RUN pip install mlflow

# Expose port
EXPOSE 5000

# Command to run mlflow server
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5050"]
