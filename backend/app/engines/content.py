import os
import sys
import chromadb
from pathlib import Path
from typing import List, Dict
from sqlalchemy.orm import Session
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.database import Movie, Rating


class ContentEngine:
	"""
	Content-based filtering using vector similarity
	Finds similar movies in content (plot, genres, etc.)
	"""

	def __init__(self) -> None:
		self.client = chromadb.PersistentClient(path="/opt/flix_flow/vectordb")
		self.collection = self.client.get_collection("movies")
		self.model = SentenceTransformer("all-MiniLM-L6-v2")
		print("Content engine loaded")

	def find_similar_to_movie(self, movie_id: int, top_n: int = 10) -> List[Dict]:
		"""
		Find movies similar to the given movie

		Args:
				movie_id: target movie ID
				top_n: Number of similar movies to return

		Returns:
				List of similar movies with similarity scores
		"""

		results = self.collection.get(
			ids=[str(movie_id)],
			include=["embeddings", "metadatas"],
		)

		if not results["ids"]:
			return []

		similar = self.collection.query(
			query_embeddings=[results["embeddings"][0]],
			n_results=top_n + 1,
			include=["metadatas", "distances"],
		)

		similar_movies = []
		for i, movie_id_result in enumerate(similar["ids"][0]):
			if movie_id_result != str(movie_id):

				metadata = similar["metadatas"][0][i]
				distance = similar["distances"][0][i]

				similar_movies.append(
					{
						"movieId": metadata["movieId"],
						"tmdbId": metadata.get("tmdbId"),
						"title": metadata["title"],
						"genres": metadata["genres"],
						"similarity_score": 1 - distance,
						"avg_rating": metadata.get("avg_rating", 0.0),
						"vote_count": metadata.get("vote_count", 0),
						"source": "content-similarity",
					}
				)

		return similar_movies[:top_n]

	def find_similar_to_user_preferences(
		self, user_id: int, db: Session, top_n: int = 20
	) -> List[Dict]:
		"""
		Find movies similar to user's liked movies

		Strategy:
		1. Get user's top-rated movies (rating >= 4.0)
		2. Find movies similar to each
		3. Aggregate and rank by frequency + similarity

		Args:
				user_id: User ID
				db: Database session
				top_n: Number of recommendations

		Returns:
				List of recommended movies
		"""

		liked_movies = (
			db.query(Rating)
			.filter(Rating.userId == user_id, Rating.rating >= 4.0)
			.all()
		)

		if not liked_movies:
			return []

		similar_movies = {}
		for rating in liked_movies:
			similar = self.find_similar_to_movie(rating.movieId, top_n=10)

			for movie in similar:
				movie_id = movie["movieId"]

				if movie_id in similar_movies:
					# Already seen - increment score and count
					similar_movies[movie_id]["score"] += movie["similarity_score"]
					similar_movies[movie_id]["count"] += 1
				else:
					similar_movies[movie_id] = {
						**movie,
						"score": movie["similarity_score"],
						"count": 1,
					}

		recommendations = list(similar_movies.values())
		recommendations.sort(key=lambda x: (x["count"], x["score"]), reverse=True)

		return recommendations[:top_n]

	def search_by_text(self, query_text: str, top_n: int = 10) -> List[Dict]:
		"""
		Search moviesby natural language query

		Example: "action movies with space adventure"

		Args:
				query_text: Natural language search query
				top_n: Number of results

		Returns:
				List of matching movies
		"""

		query_embedding = self.model.encode(query_text).tolist()

		results = self.collection.query(
			query_embeddings=[query_embedding],
			n_results=top_n,
			include=["metadatas", "distances"],
		)

		matches = []
		for i in range(len(results["ids"][0])):
			metadata = results["metadatas"][0][i]
			distance = results["distances"][0][i]
			similarity = 1 - distance

			matches.append(
				{
					"movieId": int(results["ids"][0][i]),
					"tmdbId": metadata.get("tmdbId"),
					"title": metadata["title"],
					"genres": metadata["genres"],
					"similarity_score": similarity,
					"source": "content-search",
				}
			)

		return matches

	def get_hero_movie(self, user_id: int, db: Session) -> Dict:
		"""
		Get best content-based recommendation for hero section
		"""
		recommendations = self.find_similar_to_user_preferences(user_id, db, top_n=1)
		if recommendations:
			hero = recommendations[0]
			hero["confidence"] = hero["similarity_score"]
			return hero
		return None


content_engine = ContentEngine()
