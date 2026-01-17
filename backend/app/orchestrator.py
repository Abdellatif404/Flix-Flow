from database import Rating
from sqlalchemy.orm import Session
from typing import Dict

from engines.content import content_engine
from engines.statistical import statistical_engine
from engines.collaborative import collaborative_engine


class RecommendationOrchestrator:
	"""
	Smart router that decides which engines to use based on context
	"""

	def __init__(self) -> None:
		self.collaborative = collaborative_engine
		self.statistical = statistical_engine
		self.content = content_engine

	def is_new_user(self, user_id: int, db: Session) -> bool:
		"""
		Check if user is new (has few or no ratings)
		"""
		rating_count = db.query(Rating).filter(Rating.userId == user_id).count()
		return rating_count < 5

	def get_recommendations(
		self, user_id: int, db: Session, strategy: str = "hybrid", top_n: int = 20
	) -> Dict:
		"""
		Main recommendation function with smart routing

		Args:
				user_id: User ID
				db: Database session
				strategy: "hybrid", "collaborative", "content", or "statistical"
				top_n: Number of recommendations per section

		Returns:
				Dictionary with hero movie and recommendation sections
		"""

		is_new = self.is_new_user(user_id, db)
		if is_new:
			return self._handle_new_user(db, top_n)

		if strategy == "collaborative":
			return self._collaborative_only(user_id, db, top_n)
		elif strategy == "content":
			return self._content_only(user_id, db, top_n)
		elif strategy == "statistical":
			return self._statistical_only(db, top_n)
		else:
			return self._hybrid_recommendations(user_id, db, top_n)

	def _handle_new_user(self, db: Session, top_n: int) -> Dict:
		"""
		Cold start: new users get trending movies
		"""
		trending = self.statistical.get_trending(db, top_n=top_n)
		hero = self.statistical.get_hero_movie(db)

		return {
			"strategy": "statistical",
			"is_new_user": True,
			"hero": hero,
			"sections": [
				{
					"title": "Trending & Highly Rated",
					"description": "Popular movies with excellent ratings",
					"movies": trending,
					"source": "statistical",
				}
			],
		}

	def _collaborative_only(self, user_id: int, db: Session, top_n: int) -> Dict:
		"""
		Collaborative filtering only
		"""
		collab_recs = self.collaborative.predict_for_user(user_id, db, top_n)
		hero = self.collaborative.get_hero_movie(user_id, db)

		return {
			"strategy": "collaborative",
			"is_new_user": False,
			"hero": hero,
			"sections": [
				{
					"title": "Personalized Picks",
					"description": "Based on your rating history",
					"movies": collab_recs,
					"source": "collaborative",
				}
			],
		}

	def _content_only(self, user_id: int, db: Session, top_n: int) -> Dict:
		"""
		Content-based filtering only
		"""
		content_recs = self.content.find_similar_to_user_preferences(user_id, db, top_n)
		hero = self.content.get_hero_movie(user_id, db)

		return {
			"strategy": "content",
			"is_new_user": False,
			"hero": hero,
			"sections": [
				{
					"title": "Similar to Movies You Loved",
					"description": "Based on content similarity",
					"movies": content_recs,
					"source": "content",
				}
			],
		}

	def _statistical_only(self, db: Session, top_n: int) -> Dict:
		"""
		Statistical ranking only
		"""
		trending = self.statistical.get_trending(db, top_n)
		hero = self.statistical.get_hero_movie(db)

		return {
			"strategy": "statistical",
			"is_new_user": False,
			"hero": hero,
			"sections": [
				{
					"title": "Top Rated Movies",
					"description": "Highest quality by statistical significance",
					"movies": trending,
					"source": "statistical",
				}
			],
		}

	def _hybrid_recommendations(self, user_id: int, db: Session, top_n: int) -> Dict:
		"""
		Hybrid: combine all three engines with weighted ranking
		"""
		from app.ranker import ranker

		collab_recs = self.collaborative.predict_for_user(user_id, db, top_n)
		content_recs = self.content.find_similar_to_user_preferences(user_id, db, top_n)
		stats_recs = self.statistical.get_trending(db, top_n)

		hybrid_recs = ranker.merge_and_rank(
			collab_recs,
			content_recs,
			stats_recs,
			weights=[0.5, 0.3, 0.2],
		)

		hero = self.collaborative.get_hero_movie(user_id, db)

		return {
			"strategy": "hybrid",
			"is_new_user": False,
			"hero": hero,
			"sections": [
				{
					"title": "Personalized for You",
					"description": "Hybrid recommendations combining all signals",
					"movies": hybrid_recs[:top_n],
					"source": "hybrid",
				},
				{
					"title": "Similar to Your Favorites",
					"description": "Based on content you enjoyed",
					"movies": content_recs[:top_n],
					"source": "content",
				},
				{
					"title": "Trending Now",
					"description": "Popular and highly rated",
					"movies": stats_recs[:top_n],
					"source": "statistical",
				},
			],
		}


recommendation_orchestrator = RecommendationOrchestrator()
