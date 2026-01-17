#!/bin/bash
set -e

if [ ! -f /opt/flix_flow/data/movies.db ]; then
	echo "Database not found, Loading data..."
	python3 scripts/load_data.py
else
	echo "Database found, skipping data load."
fi

if [ ! -f /opt/flix_flow/models/svd_model.pkl ]; then
	echo "SVD model not found, training model..."
	python3 scripts/train_svd.py
else
	echo "SVD model found, skipping training."
fi

VDB_PATH="/opt/flix_flow/vectordb"

if [ ! -f "$VDB_PATH/chroma.sqlite3" ]; then
	echo "No vector index found, indexing movies..."
	python3 scripts/index_movies_chromadb.py
else
	echo "Vector database already indexed, skipping."
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
