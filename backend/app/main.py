from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, Query

from database import get_db, Movie, Rating, User
from orchestrator import recommendation_orchestrator
from engines.collaborative import collaborative_engine
from engines.statistical import statistical_engine
from engines.content import content_engine

app = FastAPI(
	title="Flix Flow API",
	description="Hybrid recommendation system with SVD, Bayesian ranking, and vector search",
)

origins = [
	"http://localhost:3040",
	"http://127.0.0.1:3040",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class MovieRating(BaseModel):
	movieId: int
	rating: Optional[float] = None

	class Config:
		from_attributes = True


class OnboardRequest(BaseModel):
	ratings: List[MovieRating]


class RecommendationResponse(BaseModel):
	strategy: str
	is_new_user: bool
	hero: Optional[dict]
	sections: List[dict]

	class Config:
		from_attributes = True


@app.get("/")
async def root():
	return {
		"message": "Movie Recommendation API",
		"engines": {
			"collaborative": "SVD-based collaborative filtering",
			"statistical": "Bayesian average ranking",
			"content": "Content-based vector similarity",
		},
		"endpoints": [
			"/recommendations/{user_id}",
			"/movies/{movie_id}",
			"/movies/{movie_id}/similar",
			"/trending",
			"/search",
		],
	}


@app.post("/onboard")
async def onboard_user(request: OnboardRequest, db: Session = Depends(get_db)):
	"""
	Onboard a new user by collecting initial ratings

	User can skip movies (rating=None) or rate them (0.5-5.0)
	"""
	import time

	max_user_id = db.query(User.userId).order_by(User.userId.desc()).first()
	new_user_id = (max_user_id[0] + 1) if max_user_id else 1

	new_user = User(userId=new_user_id, created_at=int(time.time()), rating_count=0)
	db.add(new_user)

	rating_count = 0
	for movie_rating in request.ratings:
		if movie_rating.rating:
			rating = Rating(
				userId=new_user_id,
				movieId=movie_rating.movieId,
				rating=movie_rating.rating,
				timestamp=int(time.time()),
			)
			db.add(rating)
			rating_count += 1

	new_user.rating_count = rating_count
	db.commit()

	return {
		"userId": new_user_id,
		"ratings_collected": rating_count,
		"is_new_user": rating_count < 5,
		"message": "User onboarded successfully",
	}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
	user_id: int,
	strategy: str = Query(
		"hybrid", enum=["hybrid", "collaborative", "content", "statistical"]
	),
	top_n: int = Query(20, ge=1, le=50),
	db: Session = Depends(get_db),
):
	"""
	Get personalized recommendations for a user

	Strategies:
	- hybrid: Combines all engines (best results)
	- collaborative: SVD-based collaborative filtering
	- content: Vector similarity based on liked movies
	- statistical: Trending and highly-rated movies
	"""

	user = db.query(User).filter(User.userId == user_id).first()
	if not user:
		raise HTTPException(status_code=404, detail="User not found")

	result = recommendation_orchestrator.get_recommendations(
		user_id, db, strategy, top_n
	)
	return result


@app.get("/movies/{movie_id}")
async def get_movie(movie_id: int, db: Session = Depends(get_db)):
	"""
	Get details for a specific movie
	"""
	movie = db.query(Movie).filter(Movie.movieId == movie_id).first()
	if not movie:
		raise HTTPException(status_code=404, detail="Movie not found")

	return {
		"movieId": movie.movieId,
		"tmdbId": movie.tmdbId,
		"title": movie.title,
		"genres": movie.genres,
		"avg_rating": movie.avg_rating,
		"vote_count": movie.vote_count,
		"release_year": movie.release_year,
	}


@app.get("/movies/{movie_id}/similar")
async def get_similar_movies(
	movie_id: int, top_n: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)
):
	"""
	Get movies similar to a specific movie (content-based)
	"""
	movie = db.query(Movie).filter(Movie.movieId == movie_id).first()
	if not movie:
		raise HTTPException(status_code=404, detail="Movie not found")

	similar = content_engine.find_similar_to_movie(movie_id, top_n)
	return {"movieId": movie_id, "title": movie.title, "similar_movies": similar}


@app.get("/trending")
async def get_trending(
	top_n: int = Query(20, ge=1, le=50),
	genre: Optional[str] = None,
	db: Session = Depends(get_db),
):
	"""
	Get trending/highly-rated movies

	Optionally filter by genre
	"""
	if genre:
		trending = statistical_engine.get_by_genre(db, genre, top_n)
	else:
		trending = statistical_engine.get_trending(db, top_n)

	return {"genre": genre or "all", "count": len(trending), "movies": trending}


@app.get("/search")
async def search_movies(
	query: str = Query(..., min_length=1), top_n: int = Query(10, ge=1, le=50)
):
	"""
	Search movies by natural language query
	"""
	results = content_engine.search_by_text(query, top_n)

	return {"query": query, "count": len(results), "results": results}


@app.post("/rate")
async def rate_movie(
	user_id: int,
	movie_id: int,
	rating: float = Query(..., ge=0.5, le=5.0),
	db: Session = Depends(get_db),
):
	"""
	Add a new rating for a user
	"""
	import time

	existing = (
		db.query(Rating)
		.filter(Rating.userId == user_id, Rating.movieId == movie_id)
		.first()
	)

	if existing:
		existing.rating = rating
		existing.timestamp = int(time.time())
	else:
		new_rating = Rating(
			userId=user_id, movieId=movie_id, rating=rating, timestamp=int(time.time())
		)
		db.add(new_rating)

	db.commit()

	return {
		"userId": user_id,
		"movieId": movie_id,
		"rating": rating,
		"message": "Rating saved successfully",
	}


@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {
		"status": "healthy",
		"engines": {
			"collaborative": collaborative_engine.model is not None,
			"statistical": True,
			"content": content_engine.collection is not None,
		},
	}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
