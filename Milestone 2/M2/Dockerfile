FROM python:3.11-slim


# Install PostgreSQL development packages
RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# COPY app/src /app/src
# COPY app/data /app/data

ENTRYPOINT ["python", "src/main.py"]

