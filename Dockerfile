# start from an official Python 3.11 slim image
FROM python:3.11-slim

# set the working directory inside the container
WORKDIR /app

# install Tesseract OCR and its dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first — Docker caches this layer
COPY requirements.txt .

# install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app
COPY . .

# expose the port Streamlit runs on
EXPOSE 8501

# command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]