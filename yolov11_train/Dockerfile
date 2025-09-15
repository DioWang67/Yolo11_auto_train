FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# 安裝基本工具 + OpenCV 依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl zip unzip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 複製需求檔案
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 預設啟動 bash
CMD ["bash"]
