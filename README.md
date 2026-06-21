# Football Player Analytics ML

A football analytics portfolio project combining **unsupervised scouting models** and **supervised rating prediction**.

The repository contains two complementary machine learning workflows:

1. **Role-Based Scouting Clustering**  
   Uses KMeans clustering, PCA visualizations and role-profile analysis to identify football player archetypes.

2. **FIFA Overall Rating Prediction**  
   Uses position-specific XGBoost regressors to predict FIFA overall ratings by combining real player performance statistics with FIFA attribute data.

---

## Project Highlights

- Position-specific modelling for goalkeepers, defenders, midfielders and attackers
- KMeans clustering for role discovery
- PCA 2D and 3D visualizations
- Normalized cluster heatmaps for role interpretation
- Role-based scouting scores
- Fuzzy name matching across football datasets
- XGBoost regression models for FIFA overall prediction
- Actual vs predicted plots
- Feature-importance analysis
- Clean GitHub-ready project structure

---

## Repository Structure

```text
football-player-analytics-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── run_all.py
├── run_scouting.py
├── run_rating_prediction.py
│
├── src/
│   ├── __init__.py
│   ├── scouting_clustering.py
│   └── rating_prediction.py
│
├── data/
│   ├── README.md
│   │
│   ├── scouting/
│   │   └── 2022-2023 Football Player Stats.csv
│   │
│   └── rating_prediction/
│       ├── players.csv
│       ├── appearances.csv
│       └── male_players.csv
│
└── outputs/
    ├── scouting/
    │   ├── plots/
    │   └── tables/
    │
    └── rating_prediction/
        ├── plots/
        └── tables/
```

Raw datasets are **not included** in this repository because they may have licensing restrictions.

---

## 1. Role-Based Scouting Clustering

This module applies unsupervised learning to football player statistics from the 2022/23 season.

Players are first grouped by broad position:

- Goalkeeper
- Defender
- Midfielder
- Attacker

Then each group is clustered using position-specific features.

Example role archetypes:

### Goalkeepers

- Modern Sweeper Keeper
- Shot Stopper Specialist

### Defenders

- Balanced Defender
- Aggressive Ball Winner
- Defensive Destroyer

### Midfielders

- Creative Deep Playmaker
- Defensive/Box-to-Box Mid

### Attackers

- Complete Striker
- Creative Winger
- Elite Poacher
- Low Impact Forward
- Inactive/Unknown Role

The module exports:

```text
outputs/scouting/plots/
outputs/scouting/tables/
```

including PCA plots, cluster heatmaps, scouting tables and summary metrics.

---

## 2. FIFA Overall Rating Prediction

This module builds supervised machine learning models for FIFA overall rating prediction.

It combines:

- player appearance statistics
- goals, assists, minutes and appearances
- per-90 metrics
- age
- FIFA technical attributes
- fuzzy name matching between datasets

Separate XGBoost models are trained for:

- Attackers
- Midfielders
- Defenders
- Goalkeepers

The module exports:

```text
outputs/rating_prediction/plots/
outputs/rating_prediction/tables/
```

including actual vs predicted plots, feature importances, predictions and model summary tables.

---

## Dataset Setup

Create the following files locally.

### Scouting Clustering Dataset

Place this file here:

```text
data/scouting/2022-2023 Football Player Stats.csv
```

Required columns include:

```text
Player, Pos
```

plus the performance-stat columns used in `src/scouting_clustering.py`.

### Rating Prediction Datasets

Place these files here:

```text
data/rating_prediction/players.csv
data/rating_prediction/appearances.csv
data/rating_prediction/male_players.csv
```

The `players.csv` file should include:

```text
player_id, name, position, date_of_birth
```

The `appearances.csv` file should include:

```text
player_id, date, goals, assists, minutes_played, game_id
```

The `male_players.csv` file should include FIFA player attributes such as:

```text
long_name, overall, player_positions, shooting, passing, pace, ...
```

---

## Installation

From the project root, run:

```bash
python -m pip install -r requirements.txt
```

If you are using Anaconda on Windows and `python` is not recognized, run with your full Python path, for example:

```powershell
& "C:\Users\pante\anaconda3\python.exe" -m pip install -r requirements.txt
```

---

## How to Run

### Run only the scouting clustering module

```bash
python run_scouting.py
```

### Run only the rating prediction module

```bash
python run_rating_prediction.py
```

### Run both modules

```bash
python run_all.py
```

For Anaconda on Windows:

```powershell
& "C:\Users\pante\anaconda3\python.exe" run_scouting.py
& "C:\Users\pante\anaconda3\python.exe" run_rating_prediction.py
```

---

## Outputs

After running the scripts, the repository will generate plots and CSV tables.

### Scouting outputs

```text
outputs/scouting/plots/
outputs/scouting/tables/
```

Examples:

- PCA cluster visualizations
- 3D PCA plots
- normalized cluster-profile heatmaps
- top scouting players by role
- scouting summary table

### Rating prediction outputs

```text
outputs/rating_prediction/plots/
outputs/rating_prediction/tables/
```

Examples:

- actual vs predicted plots
- feature-importance charts
- prediction tables
- rating-prediction summary table

---

## Methodology Notes

The scouting module is exploratory. KMeans clusters should be interpreted as **player archetypes**, not exact player classifications.

The rating-prediction module predicts FIFA overall ratings using a mixture of real player performance data and FIFA attribute data. Because FIFA attributes are related to the target variable, the model should be interpreted as a rating-modelling exercise rather than a pure performance-only scouting model.

---

## Suggested CV Description

```text
Football Analytics & Scouting Models | Python, XGBoost, KMeans, PCA, Pandas
Built football analytics models for player rating prediction and role-based scouting; trained position-specific XGBoost models and developed KMeans/PCA clustering to identify player archetypes for scouting analysis.
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- RapidFuzz
- Matplotlib
- Seaborn

---

## Author

Markos Pantelis
