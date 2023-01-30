docker run -p 8000:8000 -p 8001:8001 --gpus device=0 \
  -v ${PWD}:/models \
  nvcr.io/nvidia/tritonserver:22.12-py3 \
  tritonserver --model-repository=/models --exit-on-error=false
