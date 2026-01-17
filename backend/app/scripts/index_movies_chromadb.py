import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from tqdm import tqdm
from chromadb.config import Settings
from app.database import SessionLocal, Movie
from sentence_transformers import SentenceTransformer


def index_movies():
	"""
	Create vector embeddings for all movies and store in ChromaDB
	"""

	print("Loading embedding model...")
	model = SentenceTransformer("all-MiniLM-L6-v2")

	print("Initializing ChromaDB...")
	client = chromadb.PersistentClient(path="/opt/flix_flow/vectordb")

	try:
		client.delete_collection("movies")
	except:
		pass

	collection = client.create_collection(
		name="movies", metadata={"description": "Movie content embeddings"}
	)

	print("Loading movies from database...")
	db = SessionLocal()
	movies = db.query(Movie).filter(Movie.vote_count >= 100).all()
	print(f"Indexing {len(movies)} movies...")

	batch_size = 100
	for i in tqdm(range(0, len(movies), batch_size)):
		batch = movies[i : i + batch_size]

		texts = []
		metadatas = []
		ids = []

		for movie in batch:
			text = f"{movie.title} {movie.genres} {movie.plot_summary or ''}"

			texts.append(text)
			ids.append(str(movie.movieId))
			metadatas.append(
				{
					"movieId": movie.movieId,
					"title": movie.title,
					"genres": movie.genres,
					"avg_rating": movie.avg_rating,
					"vote_count": movie.vote_count,
				}
			)

		embeddings = model.encode(texts).tolist()

		collection.add(
			ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
		)

	db.close()
	print(f"Indexed {len(movies)} movies into ChromaDB.")


if __name__ == "__main__":
	index_movies()
