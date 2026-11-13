import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import xgboost

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

RESULT_DIR = PROJECT_DIR / "outputs" / "results"
GRAPH_DIR = PROJECT_DIR / "outputs" / "graphs"
MODEL_DIR = PROJECT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("All libraries imported successfully!")

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("XGBoost version:", xgboost.__version__)

from pathlib import Path

file_name = "nepal_flood_weather_dataset_kaggle_2023_2026.csv"

possible_paths = [
    Path("data") / file_name,
    Path("../data") / file_name
]

DATA_PATH = next((p for p in possible_paths if p.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        f"Dataset not found. Please check that {file_name} is inside your data folder."
    )

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print(df.head())

print("Dataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nNumber of Locations:")
print(df["location"].nunique())

print("\nLocations:")
print(df["location"].unique())


print("Missing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nExact Duplicate Rows:")
print(df.duplicated().sum())

print("\nDuplicate Date-Location-River Records:")
print(df.duplicated(
    subset=["date", "location", "river"]
).sum())

print("\nZero River Discharge Values:")
print((df["river_discharge_m3s"] == 0).sum())

print("\nNegative River Discharge Values:")
print((df["river_discharge_m3s"] < 0).sum())

print("\nRiver Discharge Statistics:")
print(df["river_discharge_m3s"].describe())

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(
    subset=["date", "location", "river", "river_discharge_m3s"]
).copy()

df = df.sort_values(
    ["location", "river", "date"]
).reset_index(drop=True)

print("Preprocessing completed!")

print("\nDataset shape after preprocessing:")
print(df.shape)

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print(df.head())



from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


GRAPH_DIR = Path(__file__).resolve().parent / "outputs" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


sns.set_theme(style="whitegrid", context="notebook")


plt.figure(figsize=(12, 6))


sns.histplot(
    data=df,
    x="river_discharge_m3s",
    bins=60,
    kde=True,
    color="steelblue",
    edgecolor="white",
    alpha=0.8,
    line_kws={"linewidth": 2}
)


plt.title(
    "Distribution of River Discharge Across Monitoring Locations",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("River Discharge (m³/s)", fontsize=12)
plt.ylabel("Number of Observations", fontsize=12)


plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.grid(axis="x", visible=False)


plt.tight_layout()

output_path = GRAPH_DIR / "01_discharge_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")


sns.set_theme(style="whitegrid", context="notebook")

log_discharge = np.log1p(df["river_discharge_m3s"])


plt.figure(figsize=(12, 6))


sns.histplot(
    x=log_discharge,
    bins=60,
    kde=True,
    color="darkorange",
    edgecolor="white",
    alpha=0.8,
    line_kws={"linewidth": 2}
)


plt.title(
    "Log-Transformed Distribution of River Discharge",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Log-Transformed River Discharge (log1p m³/s)", fontsize=12)
plt.ylabel("Number of Observations", fontsize=12)


plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.grid(axis="x", visible=False)


plt.tight_layout()


output_path = GRAPH_DIR / "02_log_discharge_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")




sns.set_theme(style="whitegrid", context="notebook")


plt.figure(figsize=(12, 6))


sns.scatterplot(
    data=df,
    x="precipitation_mm",
    y="river_discharge_m3s",
    alpha=0.45,
    s=35,
    color="teal",
    edgecolor=None
)


plt.title(
    "Relationship Between Precipitation and River Discharge",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Precipitation (mm)", fontsize=12)
plt.ylabel("River Discharge (m³/s)", fontsize=12)


plt.grid(axis="both", linestyle="--", alpha=0.4)


plt.tight_layout()


output_path = GRAPH_DIR / "03_rainfall_vs_discharge.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")




sns.set_theme(style="whitegrid", context="notebook")

plt.figure(figsize=(12, 6))


sns.boxplot(
    data=df,
    x="basin",
    y="river_discharge_m3s",
    color="skyblue",
    width=0.6,
    linewidth=1.2,
    showfliers=False
)


plt.title(
    "Distribution of River Discharge Across River Basins",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("River Basin", fontsize=12)
plt.ylabel("River Discharge (m³/s)", fontsize=12)


plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.grid(axis="x", visible=False)


plt.tight_layout()


output_path = GRAPH_DIR / "04_discharge_by_basin.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")


numeric_df = df.select_dtypes(include=np.number)


correlation_matrix = numeric_df.corr()

sns.set_theme(style="white", context="notebook")


plt.figure(figsize=(16, 11))


sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    center=0,
    square=True,
    annot_kws={"size": 9},
    cbar_kws={
        "label": "Correlation Coefficient",
        "shrink": 0.8
    }
)


plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=17,
    fontweight="bold",
    pad=20
)


plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)


plt.tight_layout()


output_path = GRAPH_DIR / "05_correlation_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]


df["month_name"] = df["date"].dt.month_name()


monthly_rainfall = (
    df.groupby("month_name")["precipitation_mm"]
    .mean()
    .reindex(months)
)


sns.set_theme(style="whitegrid", context="notebook")

plt.figure(figsize=(13, 6))

ax = sns.barplot(
    x=monthly_rainfall.index,
    y=monthly_rainfall.values,
    color="steelblue",
    edgecolor="white",
    width=0.7
)


for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f",
        padding=3,
        fontsize=9
    )


plt.title(
    "Average Monthly Precipitation Across Monitoring Locations",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Precipitation (mm)", fontsize=12)


plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.grid(axis="x", visible=False)


plt.tight_layout()


output_path = GRAPH_DIR / "06_monthly_rainfall.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


plt.show()
plt.close()

print(f"Graph successfully saved to: {output_path}")



print("\n========== FEATURE ENGINEERING ==========")


df = df.sort_values(
    by=["location", "river", "date"]
).reset_index(drop=True)


df["target_discharge"] = (
    df.groupby(["location", "river"])["river_discharge_m3s"]
    .shift(-1)
)


df["target_date"] = (
    df.groupby(["location", "river"])["date"]
    .shift(-1)
)

print("\nNext-day target columns created successfully!")

print(
    df[
        [
            "date",
            "location",
            "river",
            "river_discharge_m3s",
            "target_date",
            "target_discharge"
        ]
    ].head(10)
)


df["date_difference"] = (
    df["target_date"] - df["date"]
).dt.days


df = df[
    df["date_difference"] == 1
].copy()


df.drop(columns=["date_difference"], inplace=True)


df.reset_index(drop=True, inplace=True)

print("\nConsecutive daily records retained successfully!")

print("Dataset shape after feature engineering:", df.shape)

print("\nFirst 10 records:")
print(
    df[
        [
            "date",
            "target_date",
            "location",
            "river",
            "river_discharge_m3s",
            "target_discharge"
        ]
    ].head(10)
)



df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month


df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

print("\nSeasonal features created successfully!")

print(
    df[
        [
            "date",
            "day_of_year",
            "month",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos"
        ]
    ].head(10)
)


print("\n========== FEATURE ENGINEERING SUMMARY ==========")

print("\nDataset shape:")
print(df.shape)

print("\nNew features:")
print([
    "target_discharge",
    "target_date",
    "day_of_year",
    "month",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
])

print("\nMissing values in prediction target:")
print(df["target_discharge"].isnull().sum())

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print("\nTarget date range:")
print("Start:", df["target_date"].min())
print("End:", df["target_date"].max())

print("\nFeature engineering completed successfully!")


numerical_features = [
    "latitude",
    "longitude",
    "elevation_m",
    "precipitation_mm",
    "soil_moisture_0_100cm_m3m3",
    "rain_mm",
    "temperature_mean_c",
    "dew_point_mean_c",
    "precipitation_hours",
    "relative_humidity_mean_pct",
    "wind_speed_max_kmh",
    "wind_gusts_max_kmh",
    "wind_direction_dominant_deg",
    "river_discharge_m3s",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
]


categorical_features = [
    "location",
    "river",
    "basin"
]


all_features = numerical_features + categorical_features


X = df[all_features].copy()


y = df["target_discharge"].copy()


print("\n========== FEATURE SELECTION SUMMARY ==========")

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
print("Total input features:", len(all_features))

print("\nInput dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

print("\nSelected target: Next-Day River Discharge")

print("\nFirst five input records:")
print(X.head())

print("\nFirst five target values:")
print(y.head())

print("\nMissing values in input features:", X.isnull().sum().sum())
print("Missing values in target:", y.isnull().sum())

print("\nTarget statistics:")
print(y.describe())

print("\nFeature selection completed successfully!")



train_end_date = "2024-06-30"
validation_start_date = "2024-07-01"
validation_end_date = "2024-12-31"
test_start_date = "2025-01-01"


train_mask = df["target_date"] <= train_end_date

validation_mask = (
    (df["target_date"] >= validation_start_date) &
    (df["target_date"] <= validation_end_date)
)

test_mask = df["target_date"] >= test_start_date


X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_val = X.loc[validation_mask].copy()
y_val = y.loc[validation_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()


print("\n========== DATASET SPLITTING SUMMARY ==========")

print("\nTraining Set:")
print("Features:", X_train.shape)
print("Target:", y_train.shape)
print("Target date range:",
      df.loc[train_mask, "target_date"].min(),
      "to",
      df.loc[train_mask, "target_date"].max())

print("\nValidation Set:")
print("Features:", X_val.shape)
print("Target:", y_val.shape)
print("Target date range:",
      df.loc[validation_mask, "target_date"].min(),
      "to",
      df.loc[validation_mask, "target_date"].max())

print("\nTesting Set:")
print("Features:", X_test.shape)
print("Target:", y_test.shape)
print("Target date range:",
      df.loc[test_mask, "target_date"].min(),
      "to",
      df.loc[test_mask, "target_date"].max())

print("\nTotal records:",
      len(X_train) + len(X_val) + len(X_test))


assert len(X_train) > 0, "Training set is empty!"
assert len(X_val) > 0, "Validation set is empty!"
assert len(X_test) > 0, "Testing set is empty!"

assert X_train.index.intersection(X_val.index).empty
assert X_train.index.intersection(X_test.index).empty
assert X_val.index.intersection(X_test.index).empty

print("\nAll dataset splits verified successfully!")




from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\n========== PREPROCESSING PIPELINE ==========")

print("\nNumerical preprocessing:")
print(numerical_transformer)

print("\nCategorical preprocessing:")
print(categorical_transformer)

print("\nComplete preprocessing pipeline:")
print(preprocessor)

print("\nPreprocessing pipeline created successfully!")




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


median_value = y_train.median()

median_predictions = np.full(
    shape=len(y_test),
    fill_value=median_value
)


persistence_predictions = X_test["river_discharge_m3s"].values


def evaluate_baseline(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }


median_results = evaluate_baseline(
    y_test,
    median_predictions,
    "Median Baseline"
)

persistence_results = evaluate_baseline(
    y_test,
    persistence_predictions,
    "Persistence Baseline"
)


baseline_results = pd.DataFrame([
    median_results,
    persistence_results
])

print("\n========== BASELINE MODEL RESULTS ==========")

print(baseline_results.to_string(index=False))

print("\nBaseline models created and evaluated successfully!")



from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)


print("\nTraining Random Forest model...")

rf_pipeline.fit(X_train, y_train)

print("Random Forest model training completed!")


rf_val_predictions = rf_pipeline.predict(X_val)


rf_mae = mean_absolute_error(y_val, rf_val_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_val, rf_val_predictions)
)

rf_r2 = r2_score(y_val, rf_val_predictions)


print("\n========== RANDOM FOREST VALIDATION RESULTS ==========")

print(f"MAE:      {rf_mae:.4f}")
print(f"RMSE:     {rf_rmse:.4f}")
print(f"R2 Score: {rf_r2:.4f}")

print("\nRandom Forest model trained and evaluated successfully!")




from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


print("\nTraining XGBoost model...")

xgb_pipeline.fit(X_train, y_train)

print("XGBoost model training completed!")


xgb_val_predictions = xgb_pipeline.predict(X_val)


xgb_mae = mean_absolute_error(y_val, xgb_val_predictions)

xgb_rmse = np.sqrt(
    mean_squared_error(y_val, xgb_val_predictions)
)

xgb_r2 = r2_score(y_val, xgb_val_predictions)

print("\n========== XGBOOST VALIDATION RESULTS ==========")

print(f"MAE:      {xgb_mae:.4f}")
print(f"RMSE:     {xgb_rmse:.4f}")
print(f"R2 Score: {xgb_r2:.4f}")

print("\nXGBoost model trained and evaluated successfully!")




from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


rf_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [2, 5]
}


xgb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.1],
    "model__max_depth": [3, 5]
}


tuning_results = []

best_rf_rmse = float("inf")
best_xgb_rmse = float("inf")

best_rf_pipeline = None
best_xgb_pipeline = None

best_rf_params = None
best_xgb_params = None

print("\n========== RANDOM FOREST HYPERPARAMETER TUNING ==========")


for i, params in enumerate(ParameterGrid(rf_param_grid), start=1):

    model_pipeline = clone(rf_pipeline)
    model_pipeline.set_params(**params)

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    tuning_results.append({
        "Model": "Random Forest",
        "Parameters": str(params),
        "Validation RMSE": rmse
    })

    print(f"RF Configuration {i}/8 - RMSE: {rmse:.4f}")

    if rmse < best_rf_rmse:
        best_rf_rmse = rmse
        best_rf_pipeline = model_pipeline
        best_rf_params = params


print("\n========== XGBOOST HYPERPARAMETER TUNING ==========")


for i, params in enumerate(ParameterGrid(xgb_param_grid), start=1):

    model_pipeline = clone(xgb_pipeline)
    model_pipeline.set_params(**params)

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    tuning_results.append({
        "Model": "XGBoost",
        "Parameters": str(params),
        "Validation RMSE": rmse
    })

    print(f"XGBoost Configuration {i}/8 - RMSE: {rmse:.4f}")

    if rmse < best_xgb_rmse:
        best_xgb_rmse = rmse
        best_xgb_pipeline = model_pipeline
        best_xgb_params = params



tuning_results_df = pd.DataFrame(tuning_results)


tuning_results_df = tuning_results_df.sort_values(
    by="Validation RMSE"
).reset_index(drop=True)


print("\n========== HYPERPARAMETER TUNING RESULTS ==========")

print(tuning_results_df.to_string(index=False))


print("\n========== BEST RANDOM FOREST CONFIGURATION ==========")
print("Best Parameters:", best_rf_params)
print(f"Best Validation RMSE: {best_rf_rmse:.4f}")

print("\n========== BEST XGBOOST CONFIGURATION ==========")
print("Best Parameters:", best_xgb_params)
print(f"Best Validation RMSE: {best_xgb_rmse:.4f}")

print("\nTotal configurations tested:", len(tuning_results_df))

print("\nHyperparameter tuning completed successfully!")




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np


best_rf_predictions = best_rf_pipeline.predict(X_val)
best_xgb_predictions = best_xgb_pipeline.predict(X_val)


rf_mae = mean_absolute_error(y_val, best_rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_val, best_rf_predictions))
rf_r2 = r2_score(y_val, best_rf_predictions)


xgb_mae = mean_absolute_error(y_val, best_xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_val, best_xgb_predictions))
xgb_r2 = r2_score(y_val, best_xgb_predictions)


comparison_results = pd.DataFrame([
    {
        "Model": "Tuned Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2 Score": rf_r2
    },
    {
        "Model": "Tuned XGBoost",
        "MAE": xgb_mae,
        "RMSE": xgb_rmse,
        "R2 Score": xgb_r2
    }
])


print("\n========== TUNED MODEL COMPARISON ==========")

print(comparison_results.to_string(index=False))


if rf_rmse < xgb_rmse:
    best_model_name = "Random Forest"
    best_model = best_rf_pipeline
    best_model_rmse = rf_rmse
else:
    best_model_name = "XGBoost"
    best_model = best_xgb_pipeline
    best_model_rmse = xgb_rmse

print("\n========== BEST MODEL SELECTION ==========")

print("Selected Model:", best_model_name)
print(f"Validation RMSE: {best_model_rmse:.4f}")

print("\nTuned model comparison completed successfully!")




import pandas as pd
from sklearn.base import clone


X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

print("\n========== FINAL MODEL TRAINING ==========")

print("\nCombined Training Dataset:")
print("Input shape:", X_train_val.shape)
print("Target shape:", y_train_val.shape)


final_rf_pipeline = clone(best_rf_pipeline)


final_rf_pipeline.fit(X_train_val, y_train_val)

print("\nFinal Random Forest model trained successfully!")


final_xgb_pipeline = clone(best_xgb_pipeline)


final_xgb_pipeline.fit(X_train_val, y_train_val)

print("Final XGBoost model trained successfully!")


if best_model_name == "Random Forest":
    final_best_model = final_rf_pipeline
else:
    final_best_model = final_xgb_pipeline

print("\n========== FINAL MODEL SUMMARY ==========")

print("Selected Model:", best_model_name)
print("Selection Method: Lowest Validation RMSE")
print("Training Records:", len(X_train_val))
print("Untouched Test Records:", len(X_test))

print("\nBoth final models are ready for testing!")



import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


rf_test_predictions = final_rf_pipeline.predict(X_test)
xgb_test_predictions = final_xgb_pipeline.predict(X_test)


median_value = y_train_val.median()

median_test_predictions = np.full(len(y_test), median_value)

persistence_test_predictions = X_test["river_discharge_m3s"].values

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

#
final_results = pd.DataFrame([
    {
        "Model": "Random Forest",
        **calculate_metrics(y_test, rf_test_predictions)
    },
    {
        "Model": "XGBoost",
        **calculate_metrics(y_test, xgb_test_predictions)
    },
    {
        "Model": "Median Baseline",
        **calculate_metrics(y_test, median_test_predictions)
    },
    {
        "Model": "Persistence Baseline",
        **calculate_metrics(y_test, persistence_test_predictions)
    }
])


print("\n========== FINAL TEST RESULTS ==========")

print(final_results.to_string(index=False))


print("\n========== SELECTED MODEL ==========")

print("Selected Model:", best_model_name)
print("Selection was based on validation RMSE, not test results.")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

final_results.to_csv(
    RESULT_DIR / "final_model_evaluation.csv",
    index=False
)

print("\nFinal evaluation results saved successfully!")
print("\nFinal model evaluation completed successfully!")



import matplotlib.pyplot as plt
import numpy as np


final_predictions = final_best_model.predict(X_test)

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.5,
    label="Predicted Values"
)


minimum_value = min(y_test.min(), final_predictions.min())
maximum_value = max(y_test.max(), final_predictions.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted River Discharge")
plt.xlabel("Actual River Discharge (m³/s)")
plt.ylabel("Predicted River Discharge (m³/s)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    GRAPH_DIR / "07_actual_vs_predicted_discharge.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nActual vs predicted discharge graph saved successfully!")


import matplotlib.pyplot as plt
import numpy as np


final_predictions = final_best_model.predict(X_test)


residuals = y_test.to_numpy() - final_predictions

print("\n========== RESIDUAL ANALYSIS ==========")

print("Selected Model:", best_model_name)
print(f"Mean Residual: {np.mean(residuals):.4f}")
print(f"Mean Absolute Residual: {np.mean(np.abs(residuals)):.4f}")
print(f"Residual Standard Deviation: {np.std(residuals):.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axes[0].hist(
    residuals,
    bins=40,
    edgecolor="black",
    alpha=0.7
)

axes[0].axvline(
    0,
    linestyle="--",
    label="Zero Error"
)

axes[0].set_title("Distribution of Prediction Residuals")
axes[0].set_xlabel("Residual (m³/s)")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(alpha=0.3)


axes[1].scatter(
    final_predictions,
    residuals,
    alpha=0.5
)

axes[1].axhline(
    0,
    linestyle="--",
    label="Zero Error"
)

axes[1].set_title("Residuals vs Predicted Discharge")
axes[1].set_xlabel("Predicted River Discharge (m³/s)")
axes[1].set_ylabel("Residual (m³/s)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    GRAPH_DIR / "08_residual_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nResidual analysis graph saved successfully!")
print("\nResidual analysis completed successfully!")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


location_predictions = final_best_model.predict(X_test)

location_results = pd.DataFrame({
    "Location": df.loc[X_test.index, "location"].values,
    "Actual_Discharge": y_test.values,
    "Predicted_Discharge": location_predictions
})


performance_by_location = []

for location, group in location_results.groupby("Location"):

    actual = group["Actual_Discharge"]
    predicted = group["Predicted_Discharge"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    performance_by_location.append({
        "Location": location,
        "Records": len(group),
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


location_performance_df = pd.DataFrame(performance_by_location)

location_performance_df = location_performance_df.sort_values(
    by="RMSE",
    ascending=True
)


print("\n========== MODEL PERFORMANCE BY LOCATION ==========")

print(location_performance_df.to_string(index=False))


print("\n========== LOCATION PERFORMANCE SUMMARY ==========")

print(
    "Lowest RMSE Location:",
    location_performance_df.iloc[0]["Location"]
)

print(
    "Highest RMSE Location:",
    location_performance_df.iloc[-1]["Location"]
)


plt.figure(figsize=(12, 6))

plt.bar(
    location_performance_df["Location"],
    location_performance_df["RMSE"]
)

plt.title("Random Forest Prediction Error by River Monitoring Location")
plt.xlabel("Monitoring Location")
plt.ylabel("RMSE (m³/s)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()


GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    GRAPH_DIR / "09_model_performance_by_location.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


RESULT_DIR.mkdir(parents=True, exist_ok=True)

location_performance_df.to_csv(
    RESULT_DIR / "location_performance.csv",
    index=False
)

print("\nLocation performance results saved successfully!")
print("\nModel performance by location completed successfully!")


import pandas as pd
import matplotlib.pyplot as plt


trained_preprocessor = final_best_model.named_steps["preprocessor"]


trained_model = final_best_model.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()


importance_values = trained_model.feature_importances_


feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})


feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


top_features = feature_importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)


print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

print("\nTop 20 Important Features:")

print(
    feature_importance_df.head(20).to_string(index=False)
)


plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 20 Feature Importance - Random Forest")
plt.xlabel("Feature Importance Score")
plt.ylabel("Input Features")
plt.tight_layout()


GRAPH_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    GRAPH_DIR / "10_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


RESULT_DIR.mkdir(parents=True, exist_ok=True)

feature_importance_df.to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False
)

print("\nFeature importance graph saved successfully!")
print("Feature importance results saved successfully!")

print("\nFeature importance analysis completed successfully!")




import joblib
import json
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "outputs" / "results"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== SAVING FINAL MODELS ==========")


joblib.dump(
    final_rf_pipeline,
    MODEL_DIR / "final_random_forest.pkl"
)

print("Random Forest model saved successfully!")


joblib.dump(
    final_xgb_pipeline,
    MODEL_DIR / "final_xgboost.pkl"
)

print("XGBoost model saved successfully!")


joblib.dump(
    final_best_model,
    MODEL_DIR / "selected_best_model.pkl"
)

print("Selected best model saved successfully!")


final_results.to_csv(
    RESULT_DIR / "final_model_evaluation.csv",
    index=False
)


location_performance_df.to_csv(
    RESULT_DIR / "location_performance.csv",
    index=False
)


feature_importance_df.to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False
)

print("\nAll evaluation result files saved successfully!")


saved_predictions = pd.DataFrame({
    "Date": df.loc[X_test.index, "target_date"].values,
    "Location": df.loc[X_test.index, "location"].values,
    "River": df.loc[X_test.index, "river"].values,
    "Actual_Discharge": y_test.values,
    "Predicted_Discharge": final_best_model.predict(X_test)
})

saved_predictions.to_csv(
    RESULT_DIR / "actual_vs_predicted_results.csv",
    index=False
)


model_information = {
    "project": "Next-Day River Discharge Prediction in Nepal",
    "selected_model": best_model_name,
    "selection_method": "Lowest validation RMSE",
    "validation_rmse": float(best_model_rmse),
    "training_records": int(len(X_train_val)),
    "test_records": int(len(X_test)),
    "target": "Next-Day River Discharge",
    "evaluation_metrics": ["MAE", "RMSE", "R2 Score"]
}

with open(MODEL_DIR / "model_information.json", "w") as file:
    json.dump(model_information, file, indent=4)

print("\n========== SAVED PROJECT FILES ==========")

print("\nModels:")
for file in sorted(MODEL_DIR.iterdir()):
    if file.is_file():
        print("-", file.name)

print("\nResults:")
for file in sorted(RESULT_DIR.iterdir()):
    if file.is_file():
        print("-", file.name)

print("\nAll final models and results saved successfully!")


from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "outputs" / "results"
GRAPH_DIR = PROJECT_DIR / "outputs" / "graphs"

print("\n========== FINAL PROJECT VERIFICATION ==========")


required_models = [
    "final_random_forest.pkl",
    "final_xgboost.pkl",
    "selected_best_model.pkl",
    "model_information.json"
]


required_results = [
    "final_model_evaluation.csv",
    "location_performance.csv",
    "feature_importance.csv",
    "actual_vs_predicted_results.csv"
]


required_graphs = [
    "01_discharge_distribution.png",
    "02_log_discharge_distribution.png",
    "03_rainfall_vs_discharge.png",
    "04_discharge_by_basin.png",
    "05_correlation_heatmap.png",
    "06_monthly_rainfall.png",
    "07_actual_vs_predicted_discharge.png",
    "08_residual_analysis.png",
    "09_model_performance_by_location.png",
    "10_feature_importance.png"
]


def verify_files(folder, required_files, category):

    print(f"\nChecking {category}:")

    missing_files = []

    for filename in required_files:

        file_path = folder / filename

        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"[OK] {filename}")
        else:
            print(f"[MISSING] {filename}")
            missing_files.append(filename)

    return missing_files



missing_models = verify_files(
    MODEL_DIR, required_models, "Saved Models"
)

missing_results = verify_files(
    RESULT_DIR, required_results, "Result Files"
)

missing_graphs = verify_files(
    GRAPH_DIR, required_graphs, "Generated Graphs"
)


print("\n========== SAVED MODEL TEST ==========")

loaded_model = joblib.load(
    MODEL_DIR / "selected_best_model.pkl"
)

sample_predictions = loaded_model.predict(X_test.head(5))

print("Selected Model:", best_model_name)
print("Sample predictions from loaded model:")
print(sample_predictions)

print("\nModel loading and prediction test successful!")


print("\n========== FINAL RESULTS CHECK ==========")

saved_results = pd.read_csv(
    RESULT_DIR / "final_model_evaluation.csv"
)

print(saved_results.to_string(index=False))


total_missing = (
    len(missing_models)
    + len(missing_results)
    + len(missing_graphs)
)

print("\n========== PROJECT COMPLETION SUMMARY ==========")

print("Models checked:", len(required_models))
print("Result files checked:", len(required_results))
print("Graphs checked:", len(required_graphs))

if total_missing == 0:
    print("\nSTATUS: ALL REQUIRED FILES VERIFIED!")
    print("STATUS: SAVED MODEL PREDICTION TEST PASSED!")
    print("FINAL PROJECT VERIFICATION COMPLETED SUCCESSFULLY!")
else:
    print("\nSTATUS: SOME FILES ARE MISSING.")
    print("Please check the missing files listed above.")