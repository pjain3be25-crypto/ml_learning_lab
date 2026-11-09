# Football Expected Goals (xG) Model

A machine-learning project that estimates the probability that a shot becomes a goal using StatsBomb Open Data from the 2015/16 Premier League season. The project starts with a Logistic Regression baseline and tests whether defensive positioning and goalkeeper-position features improve shot-level predictions.

## Project overview

- **Competition:** English Premier League, 2015/16
- **Matches:** 380
- **Shots:** 9,908
- **Goals:** 988
- **Freeze-frame coverage:** 9,835 shots (99.26%)
- **Identifiable opposition goalkeeper:** 9,822 shots (99.13%)

The target is binary: `goal = 1` when the shot outcome is Goal, otherwise `goal = 0`. The model's predicted probability for the positive class is interpreted as the shot's estimated xG.

## Data source

The notebook downloads StatsBomb Open Data from the [Hudl Open Data GitHub mirror](https://github.com/hudl/open-data) (repository path: `data/competitions.json`, `data/matches/2/27.json`, and the corresponding `data/events/{match_id}.json` files).

The notebook retrieves the data at runtime, so an internet connection is required to reproduce the extraction. The extracted shot table is not required to be committed to this repository. Please retain attribution to StatsBomb and follow the terms provided by the source repository when using or redistributing its data.

## Features

### Base features

| Feature | Description |
|---|---|
| `distance_to_goal` | Euclidean distance from the shot location to the centre of the goal at `(120, 40)`, in StatsBomb pitch-coordinate units. |
| `shot_angle` | Angle subtended by the two goalposts, calculated from the shot location and posts `(120, 36)` and `(120, 44)`. Stored in radians. |
| `body_part` | Body part used for the shot. |
| `shot_type` | StatsBomb shot type. |

### Additional engineered features

| Feature | Description |
|---|---|
| `central_lane_blockers` | Count of opposition outfield players whose projection lies between the shooter and goal centre and whose perpendicular distance from the shooting line is at most `2/3` pitch-coordinate units. |
| `goal_mouth_triangle_blockers` | Count of opposition outfield players inside the triangle formed by the shooter and the two goalposts. |
| `goalkeeper_distance` | Euclidean distance from the shooter to the opposition goalkeeper in that shot's freeze frame, in pitch-coordinate units. |
| `goalkeeper_angle` | Absolute angle in degrees between the shooter-to-goal-centre vector and the shooter-to-goalkeeper vector. |

Both defensive blocker counts exclude teammates and the goalkeeper. The lane threshold is an exploratory heuristic in StatsBomb coordinate units; it is not presented as a verified physical measurement.

### Missing freeze-frame / goalkeeper observations

- 73 shots have no freeze frame.
- 8 additional frames contain only the shooting team's goalkeeper, not the opposition goalkeeper.
- 5 additional frames contain no goalkeeper position entry.
- Therefore goalkeeper features are available for 9,822 of 9,908 shots. Missing numerical values are imputed within the model pipeline using the training-set median.

## Modelling approach

The notebook compares three Logistic Regression candidates:

1. **Baseline:** distance, shot angle, body part and shot type.
2. **Enhanced:** baseline features plus both defensive blocker counts, goalkeeper distance and goalkeeper angle.
3. **Enhanced without goalkeeper distance:** the enhanced feature set excluding goalkeeper distance.

Preprocessing is performed inside a Scikit-learn pipeline:

- Numerical features: median imputation followed by standardisation.
- Categorical features: most-frequent imputation followed by one-hot encoding (`handle_unknown="ignore"`).
- Classifier: `LogisticRegression(max_iter=1000, random_state=42)`.

A single 80/20 `GroupShuffleSplit` is performed using `match_id` as the group, so shots from the same match do not appear in both partitions. This produced 7,882 training shots from 304 matches and 2,026 test shots from 76 matches, with zero overlapping matches.

## Holdout results

The recorded results for the match-grouped holdout split are:

| Model | ROC-AUC ↑ | Brier Score ↓ | Log Loss ↓ |
|---|---:|---:|---:|
| Baseline | 0.75913 | 0.08063 | 0.28523 |
| Enhanced | **0.77474** | **0.07947** | **0.28033** |
| Enhanced without goalkeeper distance | 0.76906 | 0.07964 | 0.28177 |

The enhanced model is selected because it performed best on all three recorded metrics in this comparison. Its goalkeeper-distance feature is highly correlated with distance to goal (Pearson correlation 0.965), so its coefficient should be interpreted cautiously; correlation and ablation results are documented in the notebook.

**Evaluation caveat:** the same holdout set was used to compare the candidate models and select the reported model. These figures are therefore model-selection holdout results, not an untouched final test estimate. A future iteration should reserve a separate final test set or use nested/grouped cross-validation for model selection.

## Repository structure

```text
ml_learning_lab/
├── notebooks/
│   └── inspect_xgdata.ipynb
├── models/
│   ├── xg_logistic_regression.joblib
│   └── model_metadata.json
├── README.md
├── requirements.txt
└── .gitignore
```

The model artifacts are generated by the notebook. The intended location is the repository-root `models/` directory. In the current notebook, `Path("models")` is relative to the Jupyter kernel's working directory. Start the kernel from the repository root, or adjust the save cell so it resolves the repository root explicitly (for example, move one directory up when `Path.cwd().name == "notebooks"`).

## Reproducing the project

1. Clone the repository and open it in VS Code.
2. Create and activate a virtual environment (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Open `notebooks/inspect_xgdata.ipynb`.
4. Restart the notebook kernel and run all cells from top to bottom. The extraction cells download the StatsBomb match/event data and therefore require internet access.
5. The notebook creates the fitted model and `model_metadata.json` under `models/` relative to the kernel working directory. For the recommended repository layout, make the kernel working directory the repository root, or adjust the save path as described above.

To load the saved pipeline:

```python
import joblib

model = joblib.load("models/xg_logistic_regression.joblib")
```

The pipeline expects the same raw feature columns used during training; preprocessing is included in the saved pipeline.

## Limitations and next steps

- The dataset covers one competition season only. Generalisation to other seasons, leagues and competitions has not been tested.
- The evaluation uses one match-grouped holdout split.
- The defensive lane threshold is heuristic and expressed in pitch-coordinate units.
- Goalkeeper distance is strongly correlated with shot distance, so individual coefficient interpretations are sensitive to overlap between features.
- Missing freeze-frame observations are imputed; this assumes median imputation is an acceptable MVP treatment.
- Calibration is inspected with quantile-binned calibration curves, but further calibration assessment on an independent dataset would be useful.

## Technologies

Python, Pandas, NumPy, Requests, Matplotlib, Scikit-learn, Joblib and Jupyter Notebook.
