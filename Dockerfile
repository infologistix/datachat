# --- Stage 1: build the webcomponent frontend ---
FROM node:20-slim AS frontend-build

WORKDIR /app
COPY pyproject.toml ./
WORKDIR /app/frontends/webcomponent
COPY frontends/webcomponent/package*.json ./
RUN npm ci
COPY frontends/webcomponent/ ./
RUN npm run build

# --- Stage 2: python runtime ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# psycopg2-binary + build deps for any packages without wheels
RUN groupadd -g 1000 appgroup && \
    useradd -r -u 1000 -g 1000 -m -s /bin/bash appuser
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[gemini,postgres,chromadb,fastapi,openai]" python-dotenv

COPY *.py ./
COPY --from=frontend-build /app/frontends/webcomponent/dist ./frontends/webcomponent/dist
    
USER 1000

ENV HOST=0.0.0.0 \
    PORT=8084 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

EXPOSE 8084

CMD ["python", "main.py"]
