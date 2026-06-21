"""
Football Scouting Clustering
============================

Exploratory football analytics pipeline using:
- position-specific feature engineering
- KMeans clustering
- role-profile mapping
- PCA 2D / 3D visualizations
- cluster heatmaps
- role-based scouting tables

Expected dataset:
    data/scouting/2022-2023 Football Player Stats.csv

Run:
    python run_scouting.py
"""

from __future__ import annotations

import os
from itertools import permutations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

os.environ.setdefault("OMP_NUM_THREADS", "1")

RANDOM_STATE = 42
MIN_MINUTES = 900
TOP_N_PER_ROLE = 5
SHOW_PLOTS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "scouting" / "2022-2023 Football Player Stats.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scouting"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"


ROLE_COLORS = {
    "Modern Sweeper Keeper": "#1f77b4",
    "Shot Stopper Specialist": "#ff7f0e",

    "Balanced Defender": "#2ca02c",
    "Aggressive Ball Winner": "#d62728",
    "Defensive Destroyer": "#9467bd",

    "Defensive/Box-to-Box Mid": "#8c564b",
    "Creative Deep Playmaker": "#e377c2",

    "Low Impact Forward": "#17becf",
    "Complete Striker": "#7f7f7f",
    "Creative Winger": "#bcbd22",
    "Inactive/Unknown Role": "#c0392b",
    "Elite Poacher": "#f5b041",
}


FEATURES_BY_POSITION = {
    "Goalkeeper": [
        "PasTotCmp%", "PasTotDist", "PasTotPrgDist",
        "PasShoCmp%", "PasMedCmp%", "PasLonCmp%",
        "Sw", "TB", "PasProg",
        "PasLive", "PasDead", "PasFK",
        "Touches", "TouDefPen", "TouDef3rd", "TouMid3rd",
        "AerWon", "AerLost", "AerWon%",
        "Err", "OG",
    ],
    "Defender": [
        "Tkl", "TklWon", "Blocks", "BlkSh", "BlkPass",
        "Int", "Tkl+Int", "Clr", "Err",
        "TouDef3rd", "TouDefPen",
    ],
    "Midfielder": [
        "Assists", "PasAss", "Pas3rd", "PasTotCmp%",
        "PasTotDist", "PasProg", "Carries", "CarPrgDist",
        "CarProg", "Touches", "Rec", "RecProg",
    ],
    "Attacker": [
        "Goals", "Shots", "SoT", "G/Sh", "G/SoT",
        "Assists", "PasProg", "Pas3rd", "Carries",
        "CarProg", "Off", "PKatt", "ShoPK",
    ],
}


ROLE_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "Goalkeeper": {
        "Modern Sweeper Keeper": {
            "PasTotCmp%": 1.4,
            "PasTotDist": 0.8,
            "PasTotPrgDist": 1.2,
            "PasLonCmp%": 1.0,
            "Touches": 1.0,
            "TouMid3rd": 1.1,
            "Sw": 1.2,
            "TB": 0.8,
            "Err": -1.0,
            "OG": -0.8,
        },
        "Shot Stopper Specialist": {
            "TouDefPen": 1.2,
            "TouDef3rd": 1.0,
            "AerWon%": 1.2,
            "AerWon": 0.8,
            "PasShoCmp%": 0.6,
            "Err": -1.3,
            "OG": -1.0,
        },
    },

    "Defender": {
        "Balanced Defender": {
            "Tkl": 0.8,
            "TklWon": 0.8,
            "Blocks": 0.8,
            "Int": 0.8,
            "Tkl+Int": 1.0,
            "Clr": 1.0,
            "TouDef3rd": 0.8,
            "TouDefPen": 0.7,
            "Err": -1.0,
        },
        "Aggressive Ball Winner": {
            "Tkl": 1.4,
            "TklWon": 1.4,
            "Int": 1.2,
            "Tkl+Int": 1.5,
            "Blocks": 0.6,
            "Err": -0.8,
        },
        "Defensive Destroyer": {
            "Blocks": 1.3,
            "BlkSh": 1.2,
            "BlkPass": 1.0,
            "Clr": 1.5,
            "TouDefPen": 1.2,
            "TouDef3rd": 1.0,
            "Err": -1.0,
        },
    },

    "Midfielder": {
        "Defensive/Box-to-Box Mid": {
            "Touches": 1.2,
            "Rec": 1.2,
            "Carries": 1.0,
            "CarProg": 1.0,
            "CarPrgDist": 0.9,
            "PasTotDist": 0.9,
            "PasTotCmp%": 0.8,
        },
        "Creative Deep Playmaker": {
            "PasAss": 1.3,
            "Pas3rd": 1.4,
            "PasProg": 1.4,
            "RecProg": 1.0,
            "Assists": 1.2,
            "CarProg": 0.9,
            "CarPrgDist": 0.8,
        },
    },

    "Attacker": {
        "Low Impact Forward": {
            "Goals": -1.3,
            "Shots": -1.0,
            "SoT": -1.0,
            "Assists": -0.8,
            "PasProg": -0.6,
            "CarProg": -0.6,
            "Off": 0.3,
        },
        "Complete Striker": {
            "Goals": 1.3,
            "Shots": 1.1,
            "SoT": 1.2,
            "Assists": 0.8,
            "PasProg": 0.8,
            "Carries": 0.7,
            "CarProg": 0.8,
        },
        "Creative Winger": {
            "Assists": 1.3,
            "PasProg": 1.2,
            "Pas3rd": 1.2,
            "Carries": 1.1,
            "CarProg": 1.3,
            "Shots": 0.5,
            "SoT": 0.4,
        },
        "Inactive/Unknown Role": {
            "Goals": -0.8,
            "Shots": -0.8,
            "SoT": -0.8,
            "Assists": -0.8,
            "PasProg": -0.8,
            "Carries": -0.8,
            "CarProg": -0.8,
        },
        "Elite Poacher": {
            "Goals": 1.7,
            "G/Sh": 1.3,
            "G/SoT": 1.3,
            "SoT": 1.2,
            "PKatt": 0.5,
            "ShoPK": 0.5,
            "Assists": -0.2,
        },
    },
}


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def read_football_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Put the scouting dataset inside:\n"
            "data/scouting/2022-2023 Football Player Stats.csv"
        )

    encodings = ["utf-8-sig", "cp1252", "latin1", "ISO-8859-1"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, delimiter=";")
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not read dataset with tested encodings.")


def map_position(pos: str) -> str:
    pos = str(pos).strip()

    if pos == "GK":
        return "Goalkeeper"
    if pos in ["DF", "DF-MF", "DF-FW"]:
        return "Defender"
    if pos in ["MF", "MF-DF", "MF-FW"]:
        return "Midfielder"
    if pos in ["FW", "FW-MF", "FW-DF"]:
        return "Attacker"

    return "Other"


def parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("\u2212", "-", regex=False)
    )

    cleaned = cleaned.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    cleaned = cleaned.str.replace(",", ".", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def prepare_numeric_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    prepared = df.copy()

    for feature in features:
        if feature in prepared.columns:
            prepared[feature] = parse_numeric_series(prepared[feature])

    return prepared


def available_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    existing = [feature for feature in features if feature in df.columns]
    missing = sorted(set(features) - set(existing))

    if missing:
        print(f"Missing features ignored: {missing}")

    return existing


def apply_minutes_filter(df: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    if "Min" not in df.columns:
        print("No 'Min' column found. Skipping minutes filter.")
        return df

    filtered = df.copy()
    filtered["Min"] = parse_numeric_series(filtered["Min"])

    before = len(filtered)
    filtered = filtered[filtered["Min"] >= min_minutes].copy()
    after = len(filtered)

    print(f"Minutes filter: kept {after:,} of {before:,} players with Min >= {min_minutes}.")
    return filtered


# ---------------------------------------------------------------------
# Modelling utilities
# ---------------------------------------------------------------------

def compute_silhouette_scores(x_scaled: np.ndarray, max_k: int = 6) -> Dict[int, float]:
    scores = {}
    max_possible_k = min(max_k, len(x_scaled) - 1)

    for k in range(2, max_possible_k + 1):
        labels = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=30,
        ).fit_predict(x_scaled)

        scores[k] = silhouette_score(x_scaled, labels)

    return scores


def profile_score(row: pd.Series, role_weights: Dict[str, float]) -> float:
    score = 0.0

    for feature, weight in role_weights.items():
        if feature in row.index:
            score += row[feature] * weight

    return float(score)


def assign_roles_to_clusters(
    cluster_means_normalized: pd.DataFrame,
    role_profiles: Dict[str, Dict[str, float]],
) -> Dict[int, str]:
    clusters = list(cluster_means_normalized.index)
    roles = list(role_profiles.keys())

    score_matrix = np.zeros((len(clusters), len(roles)))

    for i, cluster_id in enumerate(clusters):
        row = cluster_means_normalized.loc[cluster_id]
        for j, role in enumerate(roles):
            score_matrix[i, j] = profile_score(row, role_profiles[role])

    best_score = -np.inf
    best_mapping = {}

    for perm in permutations(range(len(roles)), len(clusters)):
        total_score = sum(score_matrix[i, role_idx] for i, role_idx in enumerate(perm))
        if total_score > best_score:
            best_score = total_score
            best_mapping = {
                clusters[i]: roles[role_idx]
                for i, role_idx in enumerate(perm)
            }

    return best_mapping


def compute_player_scores(
    df_pos: pd.DataFrame,
    features: List[str],
    role_profiles: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    scored = df_pos.copy()

    player_features_normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(scored[features]),
        columns=features,
        index=scored.index,
    )

    raw_scores = []

    for idx, row in player_features_normalized.iterrows():
        role = scored.loc[idx, "Role"]
        role_weights = role_profiles.get(role, {})
        raw_scores.append(profile_score(row, role_weights))

    scored["RawScore"] = raw_scores

    def normalize_role_scores(series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return pd.Series(100.0, index=series.index)
        return 100 * (series - series.min()) / (series.max() - series.min())

    scored["Score"] = (
        scored
        .groupby("Role")["RawScore"]
        .transform(normalize_role_scores)
        .round(2)
    )

    return scored


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def save_or_show(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def plot_pca_2d(
    x_scaled: np.ndarray,
    roles: pd.Series,
    position: str,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(x_scaled)

    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Role"] = roles.values

    palette = {
        role: ROLE_COLORS.get(role, "#999999")
        for role in sorted(plot_df["Role"].dropna().unique())
    }

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Role",
        palette=palette,
        s=70,
        edgecolor="black",
        alpha=0.85,
    )

    plt.title(f"PCA Cluster Visualization - {position}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="best", fontsize=9)

    save_or_show(output_path)


def plot_pca_3d(
    x_scaled: np.ndarray,
    roles: pd.Series,
    position: str,
    output_path: Path,
) -> None:
    pca = PCA(n_components=3)
    coords = pca.fit_transform(x_scaled)

    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    plot_df["Role"] = roles.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for role in sorted(plot_df["Role"].dropna().unique()):
        role_data = plot_df[plot_df["Role"] == role]
        ax.scatter(
            role_data["PC1"],
            role_data["PC2"],
            role_data["PC3"],
            label=role,
            color=ROLE_COLORS.get(role, "#999999"),
            s=55,
            edgecolors="black",
            alpha=0.85,
        )

    ax.set_title(f"3D PCA Clustering - {position}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)")
    ax.legend(loc="upper right", fontsize=8)

    save_or_show(output_path)


def plot_cluster_heatmap(
    cluster_means: pd.DataFrame,
    cluster_to_role: Dict[int, str],
    position: str,
    output_path: Path,
) -> None:
    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index,
    )

    normalized.index = [
        cluster_to_role.get(cluster_id, f"Cluster {cluster_id}")
        for cluster_id in normalized.index
    ]

    plt.figure(figsize=(13, 5))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar=True,
    )

    plt.title(f"{position} - Normalized Cluster Profile Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Roles / Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_or_show(output_path)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = read_football_csv(DATA_PATH)

    required_columns = {"Player", "Pos"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df["Position_Group"] = df["Pos"].apply(map_position)
    df = apply_minutes_filter(df, MIN_MINUTES)

    top_players_all = []
    summary_rows = []

    for position, raw_features in FEATURES_BY_POSITION.items():
        print(f"\n{'=' * 70}")
        print(f"Processing scouting position group: {position}")
        print(f"{'=' * 70}")

        features = available_features(df, raw_features)

        if len(features) < 3:
            print(f"Skipping {position}: not enough available features.")
            continue

        df_pos = (
            df[df["Position_Group"] == position]
            [["Player", "Position_Group"] + features]
            .copy()
        )

        df_pos = prepare_numeric_features(df_pos, features)
        df_pos = df_pos.replace([np.inf, -np.inf], np.nan)
        df_pos = df_pos.dropna(subset=features)

        if len(df_pos) < 20:
            print(f"Skipping {position}: only {len(df_pos)} usable players.")
            continue

        role_profiles = ROLE_PROFILES[position]
        n_clusters = len(role_profiles)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df_pos[features])

        silhouette_scores = compute_silhouette_scores(x_scaled, max_k=6)
        chosen_silhouette = silhouette_scores.get(n_clusters, np.nan)

        print("Silhouette scores:")
        for k, score in silhouette_scores.items():
            print(f"  k={k}: {score:.3f}")

        print(f"Chosen k for predefined role taxonomy: {n_clusters}")
        print(f"Silhouette at chosen k: {chosen_silhouette:.3f}")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=50,
        )

        df_pos["Cluster"] = kmeans.fit_predict(x_scaled)

        cluster_means = df_pos.groupby("Cluster")[features].mean()

        cluster_means_normalized = pd.DataFrame(
            MinMaxScaler().fit_transform(cluster_means),
            columns=cluster_means.columns,
            index=cluster_means.index,
        )

        cluster_to_role = assign_roles_to_clusters(
            cluster_means_normalized,
            role_profiles,
        )

        df_pos["Role"] = df_pos["Cluster"].map(cluster_to_role)

        df_pos = compute_player_scores(
            df_pos=df_pos,
            features=features,
            role_profiles=role_profiles,
        )

        top_players = (
            df_pos
            .sort_values(["Role", "Score"], ascending=[True, False])
            .groupby("Role")
            .head(TOP_N_PER_ROLE)
            .reset_index(drop=True)
        )

        top_players["Position"] = position
        top_players_all.append(top_players)

        summary_rows.append({
            "Position": position,
            "Players": len(df_pos),
            "Clusters": n_clusters,
            "Silhouette": round(float(chosen_silhouette), 3),
            "Features": len(features),
            "Roles": ", ".join(sorted(df_pos["Role"].dropna().unique())),
        })

        cluster_profile_output = cluster_means.copy()
        cluster_profile_output["Role"] = [
            cluster_to_role.get(cluster_id, f"Cluster {cluster_id}")
            for cluster_id in cluster_profile_output.index
        ]

        cluster_profile_output.to_csv(
            TABLES_DIR / f"{position.lower()}_cluster_profiles.csv",
            encoding="utf-8-sig",
        )

        plot_pca_2d(
            x_scaled=x_scaled,
            roles=df_pos["Role"],
            position=position,
            output_path=PLOTS_DIR / f"{position.lower()}_pca_2d.png",
        )

        plot_pca_3d(
            x_scaled=x_scaled,
            roles=df_pos["Role"],
            position=position,
            output_path=PLOTS_DIR / f"{position.lower()}_pca_3d.png",
        )

        plot_cluster_heatmap(
            cluster_means=cluster_means,
            cluster_to_role=cluster_to_role,
            position=position,
            output_path=PLOTS_DIR / f"{position.lower()}_cluster_heatmap.png",
        )

    if not top_players_all:
        raise RuntimeError("No scouting output generated. Check input data and column names.")

    scouting_df = pd.concat(top_players_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    scouting_columns = [
        "Player",
        "Position",
        "Role",
        "Score",
        "RawScore",
        "Cluster",
    ]

    scouting_df[scouting_columns].to_csv(
        TABLES_DIR / "top_players_scouting.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        TABLES_DIR / "scouting_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTop scouting players:")
    print(scouting_df[scouting_columns].to_string(index=False))

    print("\nScouting summary:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved scouting outputs to: {OUTPUT_DIR.resolve()}")

    return summary_df


if __name__ == "__main__":
    run_pipeline()
