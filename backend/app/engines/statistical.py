import sys
import numpy as np
from pathlib import Path
from sqlalchemy import func
from typing import List, Dict
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.database import Movie, Rating


class StatisticalEngine:
	"""
	Bayesian average ranking (IMDB formula)
	Shows movies with statistically significant high ratings
	"""

	def __init__(self) -> None:
		self.global_mean = None
		self.min_votes = 50

	def calculate_global_mean(self, db: Session):
		"""Calculate mean rating across all movies"""
		if self.global_mean is None:
			result = db.query(func.avg(Movie.avg_rating)).scalar()
			self.global_mean = result or 3.0
		return self.global_mean

	def weighted_rating(self, avg_rating: float, vote_count: int, C: float, m: int):
		"""
		Calculate Bayesian weighted rating

		Formula: WR = (v / (v + m)) * R + (m / (v + m)) * C

		Where:
		- v = number of votes for the movie
		- m = minimum votes required to be considered
		- R = average rating of the movie
		- C = mean rating across all movies

		This prevents movies with 1 perfect rating from rankint #1
		"""
		v = vote_count
		R = avg_rating

		WR = (v / (v + m)) * R + (m / (v + m)) * C
		return WR

	def get_trending(
		self, db: Session, top_n: int = 20, time_window_days: int = None
	) -> List[Dict]:
		"""
		Get trending movies with high statistical ratings

		Args:
				db: Database session
				top_n: Number of results
				time_window_days: Optional - filter by recent movies

		Returns:
				List of top-rated movies by weighted score
		"""

		C = self.calculate_global_mean(db)
		m = self.min_votes
		query = db.query(Movie).filter(Movie.vote_count >= m)

		if time_window_days:
			from datetime import datetime

			current_year = datetime.now().year
			years_back = time_window_days // 365
			min_year = current_year - years_back
			query = query.filter(Movie.release_year >= min_year)

		movies = query.all()

		results = []
		for movie in movies:
			wr = self.weighted_rating(movie.avg_rating, movie.vote_count, C, m)

			results.append(
				{
					"movieId": movie.movieId,
					"tmdbId": movie.tmdbId,
					"title": movie.title,
					"genres": movie.genres,
					"avg_rating": movie.avg_rating,
					"vote_count": movie.vote_count,
					"weighted_rating": wr,
					"source": "statistical_bayesian",
				}
			)

		results.sort(key=lambda x: x["weighted_rating"], reverse=True)
		return results[:top_n]

	def get_by_genre(self, db: Session, genre: str, top_n: int = 10) -> List[Dict]:
		"""
		Get top movies in a specific genre

		Args:
				db: Database session
				genre: Genre to filter by
				top_n: Number of results

		Returns:
				Top-related movies in that genre
		"""

		C = self.calculate_global_mean(db)
		m = self.min_votes
		movies = (
			db.query(Movie)
			.filter(Movie.genres.contains(genre), Movie.vote_count >= m)
			.all()
		)

		results = []
		for movie in movies:
			wr = self.weighted_rating(movie.avg_rating, movie.vote_count, C, m)

			results.append(
				{
					"movieId": movie.movieId,
					"tmdbId": movie.tmdbId,
					"title": movie.title,
					"genres": movie.genres,
					"weighted_rating": wr,
					"source": f"statistical_genre_{genre.lower()}",
				}
			)

		results.sort(key=lambda x: x["weighted_rating"], reverse=True)
		return results[:top_n]

	def get_hero_movie(self, db: Session) -> Dict:
		"""
		Get single highest-rated movie (for hero section)
		"""
		trending = self.get_trending(db, top_n=1)
		if trending:
			hero = trending[0]
			hero["confidence"] = min(hero["weighted_rating"] / 5.0, 1.0)
			return hero

		return None


statistical_engine = StatisticalEngine()
