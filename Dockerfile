# API + operational worker image (BRIEF4 Phase 5). Not built on the development machine
# (Windows Application Control); written for a Linux host / cloud VM.
#   docker build -t sih26078 .            docker compose up
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libeccodes0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# requirements.txt pins the CPU torch wheel index; cfgrib/eccodes enable GRIB2 input on Linux
RUN pip install -r requirements.txt && pip install cfgrib==0.9.15.0
COPY pipeline pipeline
COPY synth synth
COPY backend backend
COPY scripts scripts
COPY models models
COPY reports/*.json reports/
COPY data/real/dem data/real/dem
COPY data/real/boundary data/real/boundary
COPY data/real/clim data/real/clim
EXPOSE 8000
CMD ["python", "-m", "backend.api", "--host", "0.0.0.0", "--port", "8000"]
