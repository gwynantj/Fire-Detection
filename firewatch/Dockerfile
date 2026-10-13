# Export the fire/smoke YOLO weights to ONNX so the runtime image does not need torch
FROM python:3.11-slim AS export
ARG MODEL_URL=https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch ultralytics onnx onnxslim
WORKDIR /models
RUN curl -fsSL -o best.pt "$MODEL_URL" && yolo export model=best.pt format=onnx imgsz=640 simplify=True opset=17

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=export /models/best.onnx /models/best.onnx
COPY firewatch.py .
CMD ["python", "-u", "firewatch.py"]
