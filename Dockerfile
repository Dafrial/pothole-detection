FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY *.pt ./

# Create directories for reports and captures
RUN mkdir -p reports captures

# Set environment variables
ENV FLASK_DEBUG=false
ENV PORT=7860

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
