FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV HALOS_MODE=production
ENV DISPLAY=:0

CMD ["python", "halos-launcher.py"]