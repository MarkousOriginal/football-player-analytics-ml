"""
Run both project modules:
1. Football scouting clustering
2. FIFA overall rating prediction

Make sure all required datasets are placed in the correct data folders.
"""

from src.scouting_clustering import run_pipeline as run_scouting
from src.rating_prediction import run_pipeline as run_rating_prediction


if __name__ == "__main__":
    print("\nRunning scouting clustering pipeline...")
    run_scouting()

    print("\nRunning rating prediction pipeline...")
    run_rating_prediction()

    print("\nAll pipelines completed.")
