FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir \
    "openai>=1.55.0" \
    "anthropic>=0.40.0" \
    "python-dotenv>=1.0.1"

COPY prompts ./prompts
COPY scripts ./scripts
COPY cloud ./cloud

RUN chmod +x cloud/run_full_dataset.sh

VOLUME ["/data"]

CMD ["cloud/run_full_dataset.sh"]
