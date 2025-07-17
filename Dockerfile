FROM python:3.12.10-slim

# Set working directory
WORKDIR /app

# Update 
RUN apt-get update && apt-get install -y build-essential

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create logs folder
RUN mkdir -p /app/logs

# Start bot
CMD ["python", "main.py"]
