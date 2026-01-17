import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import re
import pandas as pd
from pathlib import Path
from app.database import engine, init_db


def extract_release_year(title):
	"""Extracts the release year from the movie title if present."""
	match = re.search(r"\((\d{4})\)", title)
	if match:
		return int(match.group(1))
	return None


def load_movies():
	"""Loads movies into database."""
	print("Loading movies...")

	df = pd.read_csv("/opt/flix_flow/data/raw/ml-32m/movies.csv")

	links_df = pd.read_csv("/opt/flix_flow/data/raw/ml-32m/links.csv")
	df["release_year"] = df["title"].apply(extract_release_year)
	df = df[(df["release_year"] >= 2020) | (df["release_year"].isna())]
	df = df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="left")

	ratings_df = pd.read_csv("/opt/flix_flow/data/raw/ml-32m/ratings.csv")

	valid_movie_ids = df["movieId"].unique()
	ratings_filtered = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)]

	stats = (
		ratings_filtered.groupby("movieId")
		.agg({"rating": ["mean", "count"]})
		.reset_index()
	)
	stats.columns = ["movieId", "avg_rating", "vote_count"]

	df = df.merge(stats, on="movieId", how="left")
	df["tmdbId"] = df["tmdbId"].fillna(0).astype(int)
	df["avg_rating"] = df["avg_rating"].fillna(0.0)
	df["vote_count"] = df["vote_count"].fillna(0).astype(int)
	df["plot_summary"] = None
	df["poster_url"] = None

	df.to_sql("movies", engine, if_exists="replace", index=False)
	print(f"{len(df)} movies loaded.")


def load_ratings():
	"""Loads ratings into database."""
	print("Loading ratings...")

	movies_df = pd.read_sql("select movieId from movies", engine)
	valid_ids = movies_df["movieId"].unique()

	df = pd.read_csv("/opt/flix_flow/data/raw/ml-32m/ratings.csv")
	df = df[df["movieId"].isin(valid_ids)]

	df.to_sql("ratings", engine, if_exists="replace", index=False)
	print(f"{len(df)} ratings loaded.")


def load_users():
	"""Create users table from ratings data."""
	print("Loading users...")

	ratings_df = pd.read_csv("/opt/flix_flow/data/raw/ml-32m/ratings.csv")

	users = (
		ratings_df.groupby("userId")
		.agg({"timestamp": "min", "rating": "count"})
		.reset_index()
	)

	users.columns = ["userId", "created_at", "rating_count"]
	users.to_sql("users", engine, if_exists="replace", index=False)
	print(f"{len(users)} users loaded.")


if __name__ == "__main__":
	print("Initializing database and loading data...")
	init_db()
	print("Database initialized.")
	load_movies()
	load_ratings()
	load_users()
	print("Data loading complete.")
