from typing import List, Dict


class Ranker:
	"""
	Merges and ranks recommendations from multiple engines
	Uses weighted scoring + deduplication
	"""

	def normalize_scores(
		self, recommendations: List[Dict], score_key: str
	) -> Dict[int, float]:
		"""
		Normalize scores to 0-1 range using min-max scaling

		Args:
				recommendations: List of recommendations
				score_key: Key containing the score ('predicted_rating', 'weighted_rating', etc.)

		Returns:
				Dictionary mapping movieId to normalized score
		"""
		if not recommendations:
			return {}

		scores = [rec[score_key] for rec in recommendations]
		min_score = min(scores)
		max_score = max(scores)

		if max_score == min_score:
			return {rec["movieId"]: 1.0 for rec in recommendations}

		normalized = {}
		for rec in recommendations:
			movie_id = rec["movieId"]
			raw_score = rec[score_key]
			norm_score = (raw_score - min_score) / (max_score - min_score)
			normalized[movie_id] = norm_score

		return normalized

	def merge_and_rank(
		self,
		collab_recs: List[Dict],
		content_recs: List[Dict],
		stats_recs: List[Dict],
		weights: List[float] = [0.5, 0.3, 0.2],
	) -> List[Dict]:
		"""
		Merge recommendations from three engines with weighted scoring

		Args:
				collab_recs: From collaborative filtering
				content_recs: From content-based filtering
				stats_recs: From statistical ranking
				weights: [collab_weight, content_weight, stats_weight] (must sum to 1.0)

		Returns:
				Merged and ranked list of recommendations
		"""
		assert abs(sum(weights) - 1.0) < 0.01, "Weights must sum to 1.0"

		collab_weight, content_weight, stats_weight = weights

		collab_normalized = self.normalize_scores(collab_recs, "predicted_rating")
		content_normalized = self.normalize_scores(content_recs, "score")
		stats_normalized = self.normalize_scores(stats_recs, "weighted_rating")

		all_movie_ids = set()
		all_movie_ids.update(collab_normalized.keys())
		all_movie_ids.update(content_normalized.keys())
		all_movie_ids.update(stats_normalized.keys())

		combined_scores = {}
		movie_details = {}

		for movie_id in all_movie_ids:
			score = 0.0

			if movie_id in collab_normalized:
				score += collab_normalized[movie_id] * collab_weight

			if movie_id in content_normalized:
				score += content_normalized[movie_id] * content_weight

			if movie_id in stats_normalized:
				score += stats_normalized[movie_id] * stats_weight

			combined_scores[movie_id] = score

			for rec in collab_recs + content_recs + stats_recs:
				if rec["movieId"] == movie_id:
					movie_details[movie_id] = {
						"movieId": movie_id,
						"tmdbId": rec.get("tmdbId"),
						"title": rec["title"],
						"genres": rec["genres"],
						"combined_score": score,
						"source": "hybrid_weighted",
					}
					break

		ranked = sorted(
			movie_details.values(), key=lambda x: x["combined_score"], reverse=True
		)
		return ranked

	def rerank_with_diversity(
		self, recommendations: List[Dict], diversity_factor: float = 0.3
	) -> List[Dict]:
		"""
		Re-rank to increase genre diversity

		Prevents all recommendations from being the same genre

		Args:
				recommendations: Initial ranking
				diversity_factor: How much to boost diversity (0-1)

		Returns:
				Diversified ranking
		"""
		if not recommendations:
			return []

		result = []
		seen_genres = set()
		remaining = recommendations.copy()

		while remaining:
			best_score = -1
			best_idx = 0

			for i, rec in enumerate(remaining):
				genres = set(rec["genres"].split("|"))

				score = rec.get("combined_score", 0)

				new_genres = genres - seen_genres
				if new_genres:
					boost = diversity_factor * len(new_genres) / len(genres)
					score += boost

				if score > best_score:
					best_score = score
					best_idx = i

			chosen = remaining.pop(best_idx)
			result.append(chosen)

			chosen_genres = set(chosen["genres"].split("|"))
			seen_genres.update(chosen_genres)

		return result


ranker = Ranker()
