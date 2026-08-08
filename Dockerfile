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

# Install the project and the optional analysis stacks through the same
# versioned package metadata used outside the container.
COPY pyproject.toml README.md ./
COPY src ./src
COPY generate_training_data.py generate_visualizations.py plot_ldf.py ./
COPY regenerate_all.py run_full_sim.py run_gamma_camera_pipeline.py train_gnn.py ./
RUN pip install --no-cache-dir ".[viz,ml,io]"

# Copy the rest of the codebase
COPY . .

# Default command (can be overridden to run train_gnn.py)
CMD ["python", "generate_training_data.py"]
