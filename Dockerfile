# 1. PyTorch wala base image uthao
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Linux ki zaruri libraries install karo (OpenCV ke liye)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 3. Work directory set karo
WORKDIR /app

# 4. Requirements copy aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod ultralytics

# 5. Saara code aur models copy karo
COPY . .

# 6. Runpod worker ko launch karo
CMD ["python", "-u", "/app/handler.py"]