#!/bin/bash
# Start the vLLM server in Docker with Qwen3-Coder-30B-A3B-AWQ

MODEL_PATH="D:/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
CONTAINER_NAME="vllm-server"

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: 8000"

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -v "$MODEL_PATH":/models/qwen3-coder \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/qwen3-coder \
  --served-model-name qwen3-coder \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

echo "Container started. Waiting for server to be ready..."
echo "Check logs with: docker logs -f $CONTAINER_NAME"
echo "Test with: curl http://localhost:8000/v1/models"
