FROM python:3.11-slim

# System dependencies required by rasterio / GDAL (pulled in by titiler.core)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgdal36 \
    libexpat1 \
    libproj25 \
    libgeos-c1t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

CMD ["sh", "entrypoint.sh"]
