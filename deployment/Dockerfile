FROM --platform=linux/amd64 python:3.9.18-slim

WORKDIR /app
RUN apt-get update
RUN apt-get -y install gcc
# Dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir  -r requirements.txt

# Make custom src code visible
COPY ./src /app/src
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Trained model and definition with main script
COPY ./saved_model /app/saved_model
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]