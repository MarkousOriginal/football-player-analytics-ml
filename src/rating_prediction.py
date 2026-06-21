"""
FIFA Overall Rating Prediction
==============================

Position-specific supervised learning pipeline using:
- player appearance statistics
- FIFA attribute data
- fuzzy name matching
- XGBoost regression
- model evaluation and feature importance plots

Expected datasets:
    data/rating_prediction/players.csv
    data/rating_prediction/appearances.csv
    data/rating_prediction/male_players.csv

Run:
    python run_rating_prediction.py
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rapidfuzz import process, fuzz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

os.environ.setdefault("OMP_NUM_THREADS", "1")

RANDOM_STATE = 42
MATCH_THRESHOLD = 90
MIN_PLAYERS_PER_GROUP = 30
TEST_SIZE = 0.20
SHOW_PLOTS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "rating_prediction"
PLAYERS_PATH = DATA_DIR / "players.csv"
APPEARANCES_PATH = DATA_DIR / "appearances.csv"
FIFA_PATH = DATA_DIR / "male_players.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rating_prediction"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"


FIFA_COLS = [
    "long_name", "overall", "player_positions",
    "attacking_finishing", "attacking_volleys",
    "movement_reactions", "pace", "shooting",
    "skill_long_passing", "mentality_vision", "mentality_positioning",
    "passing", "skill_ball_control",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "physic", "mentality_interceptions",
    "goalkeeping_diving", "goalkeeping_handling",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
]


BASE_FEATURES = [
    "goals",
    "assists",
    "goals_per90",
    "assists_per90",
    "minutes",
    "apps",
    "age",
]


FEATURES_BY_POSITION = {
    "Attacker": [
        "attacking_finishing",
        "attacking_volleys",
        "movement_reactions",
        "pace",
        "shooting",
    ],
    "Midfielder": [
        "skill_long_passing",
        "mentality_vision",
        "mentality_positioning",
        "passing",
        "skill_ball_control",
    ],
    "Defender": [
        "defending_marking_awareness",
        "defending_standing_tackle",
        "defending_sliding_tackle",
        "physic",
        "mentality_interceptions",
    ],
    "Goalkeeper": [
        "goalkeeping_diving",
        "goalkeeping_handling",
        "goalkeeping_positioning",
        "goalkeeping_reflexes",
        "goalkeeping_speed",
    ],
}


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}\n\n"
            "Place the required rating-prediction datasets inside:\n"
            "data/rating_prediction/"
        )


def clean_player_name(name: str) -> str:
    value = str(name).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]", "", value)
    return value


def read_csv_flexible(path: Path, usecols: List[str] | None = None) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "ISO-8859-1"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV file with tested encodings: {path}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(PLAYERS_PATH, "players.csv")
    require_file(APPEARANCES_PATH, "appearances.csv")
    require_file(FIFA_PATH, "male_players.csv")

    players = read_csv_flexible(PLAYERS_PATH)
    appearances = read_csv_flexible(APPEARANCES_PATH)
    fifa = read_csv_flexible(FIFA_PATH, usecols=FIFA_COLS)

    return players, appearances, fifa


def map_fifa_position(player_positions: str) -> str:
    pos = str(player_positions)

    if any(p in pos for p in ["ST", "CF", "RW", "LW", "RF", "LF"]):
        return "Attacker"
    if any(p in pos for p in ["CM", "CAM", "CDM", "RM", "LM", "LAM", "RAM", "RCM", "LCM"]):
        return "Midfielder"
    if any(p in pos for p in ["CB", "LB", "RB", "LCB", "RCB", "LWB", "RWB"]):
        return "Defender"
    if "GK" in pos:
        return "Goalkeeper"

    return "Other"


def build_performance_table(players: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    required_appearance_cols = {"player_id", "date", "goals", "assists", "minutes_played", "game_id"}
    missing_appearance = required_appearance_cols - set(appearances.columns)
    if missing_appearance:
        raise ValueError(f"appearances.csv is missing columns: {missing_appearance}")

    required_player_cols = {"player_id", "name", "position", "date_of_birth"}
    missing_player = required_player_cols - set(players.columns)
    if missing_player:
        raise ValueError(f"players.csv is missing columns: {missing_player}")

    appearances = appearances.copy()
    players = players.copy()

    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["season"] = appearances["date"].dt.year

    for col in ["goals", "assists", "minutes_played"]:
        appearances[col] = pd.to_numeric(appearances[col], errors="coerce").fillna(0)

    performance = (
        appearances
        .groupby(["player_id", "season"], as_index=False)
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes_played", "sum"),
            apps=("game_id", "count"),
        )
    )

    players["birth_year"] = pd.to_datetime(players["date_of_birth"], errors="coerce").dt.year

    performance = performance.merge(
        players[["player_id", "name", "position", "birth_year"]],
        on="player_id",
        how="left",
    )

    performance["age"] = performance["season"] - performance["birth_year"]

    agg_stats = (
        performance
        .groupby("player_id", as_index=False)
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            minutes=("minutes", "sum"),
            apps=("apps", "sum"),
            age=("age", "mean"),
            name=("name", "first"),
            position=("position", "first"),
        )
    )

    agg_stats = agg_stats[agg_stats["minutes"] > 0].copy()
    agg_stats["goals_per90"] = (agg_stats["goals"] / agg_stats["minutes"]) * 90
    agg_stats["assists_per90"] = (agg_stats["assists"] / agg_stats["minutes"]) * 90
    agg_stats["clean_name"] = agg_stats["name"].apply(clean_player_name)

    return agg_stats


def fuzzy_match_players(agg_stats: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    fifa = fifa.dropna(subset=["long_name", "overall"]).copy()
    fifa["clean_name"] = fifa["long_name"].apply(clean_player_name)

    # Drop duplicate clean names to make the fuzzy lookup deterministic.
    fifa_lookup = fifa.drop_duplicates("clean_name").set_index("clean_name")["overall"].to_dict()
    fifa_names = list(fifa_lookup.keys())

    matched_names = []
    match_scores = []
    fifa_overalls = []

    for name in agg_stats["clean_name"]:
        match = process.extractOne(name, fifa_names, scorer=fuzz.ratio)

        if match and match[1] >= MATCH_THRESHOLD:
            matched_names.append(match[0])
            match_scores.append(match[1])
            fifa_overalls.append(fifa_lookup[match[0]])
        else:
            matched_names.append(None)
            match_scores.append(np.nan)
            fifa_overalls.append(np.nan)

    matched = agg_stats.copy()
    matched["matched_name"] = matched_names
    matched["match_score"] = match_scores
    matched["fifa_overall"] = fifa_overalls

    matched = matched.dropna(subset=["fifa_overall", "matched_name"]).copy()

    fifa_extra = fifa[
        ["clean_name", "player_positions"] +
        [col for col in FIFA_COLS if col not in ["long_name", "overall", "player_positions"]]
    ].drop_duplicates("clean_name")

    matched = matched.merge(
        fifa_extra,
        left_on="matched_name",
        right_on="clean_name",
        how="left",
        suffixes=("", "_fifa"),
    )

    matched["position_group"] = matched["player_positions"].apply(map_fifa_position)

    return matched


# ---------------------------------------------------------------------
# Modelling utilities
# ---------------------------------------------------------------------

def build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_or_show(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    group: str,
    r2: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.8)

    min_val = min(float(y_test.min()), float(np.min(y_pred)))
    max_val = max(float(y_test.max()), float(np.max(y_pred)))

    plt.plot([min_val, max_val], [min_val, max_val], "--")
    plt.xlabel("Actual FIFA Overall")
    plt.ylabel("Predicted FIFA Overall")
    plt.title(f"{group}: Actual vs Predicted (R²={r2:.2f})")

    save_or_show(output_path)


def plot_feature_importance(
    model: XGBRegressor,
    features: List[str],
    group: str,
    output_path: Path,
) -> pd.DataFrame:
    importance_df = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(8, 4))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"{group}: Top 10 Feature Importances")

    save_or_show(output_path)

    return importance_df


def train_group_model(data: pd.DataFrame, group: str) -> Dict[str, float] | None:
    group_data = data[data["position_group"] == group].copy()

    if len(group_data) < MIN_PLAYERS_PER_GROUP:
        print(f"Skipping {group}: only {len(group_data)} matched players.")
        return None

    features = BASE_FEATURES + FEATURES_BY_POSITION[group]

    available = [feature for feature in features if feature in group_data.columns]
    missing = sorted(set(features) - set(available))

    if missing:
        print(f"{group} missing features ignored: {missing}")

    if len(available) < 4:
        print(f"Skipping {group}: not enough usable features.")
        return None

    X = group_data[available].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(group_data["fifa_overall"], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X) < MIN_PLAYERS_PER_GROUP:
        print(f"Skipping {group}: only {len(X)} valid labelled players.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print(f"\nPosition: {group}")
    print(f"  Players: {len(X)}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")

    plot_actual_vs_predicted(
        y_test=y_test,
        y_pred=y_pred,
        group=group,
        r2=r2,
        output_path=PLOTS_DIR / f"{group.lower()}_actual_vs_predicted.png",
    )

    importance_df = plot_feature_importance(
        model=model,
        features=available,
        group=group,
        output_path=PLOTS_DIR / f"{group.lower()}_feature_importance.png",
    )

    importance_df.to_csv(
        TABLES_DIR / f"{group.lower()}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_pred,
            "error": y_pred - y_test,
        }
    )

    predictions_df.to_csv(
        TABLES_DIR / f"{group.lower()}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "Position": group,
        "Players": len(X),
        "Features": len(available),
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3),
    }


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    players, appearances, fifa = load_data()

    performance = build_performance_table(players, appearances)
    matched = fuzzy_match_players(performance, fifa)

    matched.to_csv(
        TABLES_DIR / "matched_player_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_summary = (
        matched["position_group"]
        .value_counts()
        .rename_axis("position_group")
        .reset_index(name="matched_players")
    )

    match_summary.to_csv(
        TABLES_DIR / "matched_players_by_position.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMatched players by position:")
    print(match_summary.to_string(index=False))

    results = []

    for group in ["Attacker", "Midfielder", "Defender", "Goalkeeper"]:
        result = train_group_model(matched, group)

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No rating-prediction models were trained. Check input files and matching quality.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        TABLES_DIR / "rating_prediction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRating prediction summary:")
    print(results_df.to_string(index=False))

    print(f"\nSaved rating-prediction outputs to: {OUTPUT_DIR.resolve()}")

    return results_df


if __name__ == "__main__":
    run_pipeline()
