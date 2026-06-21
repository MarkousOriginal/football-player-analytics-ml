# Data Folder

Raw datasets are not included in this repository because they may have licensing restrictions.

## Required files

### Scouting clustering

Place the scouting dataset here:

```text
data/scouting/2022-2023 Football Player Stats.csv
```

### Rating prediction

Place the rating-prediction datasets here:

```text
data/rating_prediction/players.csv
data/rating_prediction/appearances.csv
data/rating_prediction/male_players.csv
```

## Notes

The scripts expect these exact filenames by default.

If your files have different names, update the path constants at the top of:

```text
src/scouting_clustering.py
src/rating_prediction.py
```
