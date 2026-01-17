import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split


def train_svd_model():
	"""Train SVD model on ratings data."""

	print("Loading ratings from database...")
	from app.database import engine

	ratings_df = pd.read_sql("SELECT userId, movieId, rating FROM ratings", engine)

	print(f"Training on {len(ratings_df)} ratings...")

	reader = Reader(rating_scale=(0.5, 5.0))
	data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

	trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

	print("Training SVD model...")
	svd = SVD(
		n_factors=100,
		n_epochs=20,
		lr_all=0.005,
		reg_all=0.02,
		random_state=42,
		verbose=True,
	)

	svd.fit(trainset)

	from surprise import accuracy

	predictions = svd.test(testset)
	rmse = accuracy.rmse(predictions)
	print(f"Model trained! RMSE: {rmse:.4f}")

	with open("../models/svd_model.pkl", "wb") as f:
		pickle.dump(svd, f)

	print("Model saved to models/svd_model.pkl")
	return svd


if __name__ == "__main__":
	train_svd_model()
