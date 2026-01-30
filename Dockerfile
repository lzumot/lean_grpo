FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Lean 4
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
ENV PATH="/root/.elan/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY configs/ ./configs/

# Install the package
RUN pip install -e "."

# Expose port for vLLM
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
