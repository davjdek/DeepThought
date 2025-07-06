FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze di sistema necessarie per FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip e installa tools
RUN pip install --upgrade pip setuptools wheel

# Installa prima numpy (necessario per FAISS)
RUN pip install numpy==1.24.3

# Installa FAISS da conda-forge (più affidabile)
RUN pip install --index-url https://pypi.anaconda.org/conda-forge/simple faiss-cpu==1.7.4

# Copia requirements senza FAISS
COPY requirements.txt .

# Installa le altre dipendenze
RUN pip install -r requirements.txt

# Copia il codice
COPY . .

# Esponi la porta
EXPOSE 8000

# Comando di avvio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]