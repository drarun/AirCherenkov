# Use an official PyTorch runtime with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables to prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src

# Set working directory
WORKDIR /workspace

# Install system dependencies (for h5py, pyvbf bindings, and CV2 if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Add torch-geometric and h5py which were used in the analysis pipeline
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch-geometric h5py

# Copy the rest of the codebase
COPY . .

# Default command (can be overridden to run train_gnn.py)
CMD ["python", "generate_training_data.py"]
