# Use official lightweight Python image
FROM python:3.9-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (e.g., for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching layers
COPY pyproject.toml .
# A trick to install deps from pyproject.toml without copying full source yet
# We install build tools, then install the project in editable mode later
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the project dependencies
# We copy everything now to install the package itself
COPY src/ src/
COPY configs/ configs/
COPY README.md .

# Install the package
RUN pip install --no-cache-dir .

# Expose ports for MLflow (5000) and API (8000)
EXPOSE 5000
EXPOSE 8000

# Default command: run the inference server
# Can be overridden to run training pipeline: python -m picture_tool.main_pipeline ...
CMD ["python", "-m", "picture_tool.serve"]
