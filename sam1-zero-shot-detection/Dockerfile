FROM nvcr.io/nvidia/pytorch:26.03-py3

# Install flash-attn from Jetson AI Lab (ARM64/SBSA wheel for SM121)
RUN pip install --no-cache-dir flash-attn \
    --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128

# Install SAM3 and dependencies
RUN pip install --no-cache-dir \
    einops psutil pycocotools fastapi uvicorn[standard] \
    && pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git

COPY sam3/server.py /app/server.py
WORKDIR /app

EXPOSE 8105
CMD ["python", "server.py"]
