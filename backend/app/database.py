import os
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, Text

DB_DIR = Path("/opt/flix_flow/data")
DB_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/movies.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Movie(Base):
	__tablename__ = "movies"

	movieId = Column(Integer, primary_key=True, index=True)
	tmdbId = Column(Integer, nullable=True)
	title = Column(String, index=True)
	genres = Column(String)
	plot_summary = Column(Text, nullable=True)
	avg_rating = Column(Float, default=0.0)
	vote_count = Column(Integer, default=0)
	release_year = Column(Integer, nullable=True)
	poster_url = Column(String, nullable=True)


class Rating(Base):
	__tablename__ = "ratings"

	userId = Column(Integer, primary_key=True, index=True)
	movieId = Column(Integer, primary_key=True, index=True)
	rating = Column(Float)
	timestamp = Column(Integer)


class User(Base):
	__tablename__ = "users"

	userId = Column(Integer, primary_key=True, index=True)
	created_at = Column(Integer)
	rating_count = Column(Integer, default=0)


def init_db():
	Base.metadata.create_all(bind=engine)


def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()
