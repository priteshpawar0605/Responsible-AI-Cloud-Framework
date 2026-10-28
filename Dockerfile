FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Fix permissions for entrypoint script
RUN chmod +x start.sh

# Start both services
CMD ["./start.sh"]
