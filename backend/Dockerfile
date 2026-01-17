FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/flix_flow/app

RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	unzip \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN mkdir -p /opt/flix_flow/data/raw && \
    curl -L "https://files.grouplens.org/datasets/movielens/ml-32m.zip" \
    -o /opt/flix_flow/data/movielens.zip && \
    unzip /opt/flix_flow/data/movielens.zip -d /opt/flix_flow/data/raw/ && \
    rm /opt/flix_flow/data/movielens.zip

COPY app/ .
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
