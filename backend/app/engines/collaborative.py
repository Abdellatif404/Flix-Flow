import os
import sys
import pickle
import numpy as np
from surprise import SVD
from pathlib import Path
from typing import List, Dict
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.database import Movie, Rating

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class CollaborativeEngine:
	"""
	SVD-based collaborative filtering
	Predicts ratings based on user-movie latent factors
	"""

	def __init__(self):
		self.model = None
		self.load_model()

	def load_model(self):
		"""Load trained SVD model"""
		model_path = f"{BASE_DIR}/models/svd_model.pkl"
		if os.path.exists(model_path):
			with open(model_path, "rb") as f:
				self.model = pickle.load(f)
			print("SVD model loaded")
		else:
			raise FileNotFoundError("SVD model not found. Run train_svd.py first.")

	def predict_for_user(
		self, user_id: int, db: Session, top_n: int = 20
	) -> List[Dict]:
		"""
		Generate recommendations for a user

		Args:
				user_id: User ID
				db: Database session
				top_n: Number of recommendations

		Returns:
				List of {movieId, predicted_rating, title, genres}
		"""

		rated_movies = db.query(Rating.movieId).filter(Rating.userId == user_id).all()
		rated_movie_ids = {m[0] for m in rated_movies}

		all_movies = db.query(Movie).all()

		predictions = []
		for movie in all_movies:
			if movie.movieId not in rated_movie_ids:
				pred = self.model.predict(user_id, movie.movieId)
				predictions.append(
					{
						"movieId": movie.movieId,
						"tmdbId": movie.tmdbId,
						"title": movie.title,
						"genres": movie.genres,
						"predicted_rating": pred.est,
						"source": "collaborative_svd",
					}
				)

		predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
		return predictions[:top_n]

	def predict_single(self, user_id: int, movie_id: int) -> float:
		"""
		Predict rating for a single user-movie pair

		Args:
				user_id: User ID
				movie_id: Movie ID

		Returns:
				Predicted rating (0.5-5.0)
		"""
		pred = self.model.predict(user_id, movie_id)
		return pred.est

	def get_hero_movie(self, user_id: int, db: Session) -> Dict:
		"""
		Get the single best recommendation (for hero section)

		Returns:
				Movie with highest predicted rating
		"""
		recommendations = self.predict_for_user(user_id, db, top_n=1)
		if recommendations:
			hero = recommendations[0]
			hero["confidence"] = min(hero["predicted_rating"] / 5.0, 1.0)
			return hero
		return None


collaborative_engine = CollaborativeEngine()
