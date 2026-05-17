"""Feature engineering and preprocessing for the shipment-delay dataset.

Target naming caveat: the raw column `Reached.on.Time_Y.N` equals 1 when the
shipment was DELAYED (did NOT reach on time) and 0 when it was on time. We
rename it to `delayed` internally to remove the ambiguity. The API never
exposes the target column.

The fitted Pipeline returned by `build_preprocessor()` is saved to
`model/preprocessor.pkl` and used identically at training and inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Paths (resolved relative to repo root, which is the parent of `src/`)
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "Train.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "model"

# Fixed weight normalization denominator (dataset max ≈ 7000). Using a fixed
# constant rather than a fitted scaler keeps FeatureEngineer stateless and
# avoids train/inference drift if the input distribution shifts.
WEIGHT_NORM = 7000.0

# Column groups after feature engineering
CATEGORICAL_COLS = [
    "warehouse_block",
    "mode_of_shipment",
    "product_importance",
    "gender",
    "weight_class",
]
NUMERIC_COLS = [
    "customer_care_calls",
    "customer_rating",
    "cost_of_product",
    "prior_purchases",
    "discount_offered",
    "weight_in_gms",
    "discount_x_weight",
    "calls_per_cost",
    "high_discount",
]

# Renaming map: raw CSV header -> internal snake_case names
COLUMN_RENAME = {
    "Warehouse_block": "warehouse_block",
    "Mode_of_Shipment": "mode_of_shipment",
    "Customer_care_calls": "customer_care_calls",
    "Customer_rating": "customer_rating",
    "Cost_of_the_Product": "cost_of_product",
    "Prior_purchases": "prior_purchases",
    "Product_importance": "product_importance",
    "Gender": "gender",
    "Discount_offered": "discount_offered",
    "Weight_in_gms": "weight_in_gms",
    "Reached.on.Time_Y.N": "delayed",
}


def load_raw_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load Train.csv, strip BOM, rename to snake_case, drop ID column."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_RENAME)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    # Normalize categorical string casing so 'low'/'Low'/'LOW' all map to 'low'.
    for col in ["warehouse_block", "mode_of_shipment", "product_importance", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer that adds engineered columns.

    Adds:
      - weight_class: categorical bin (light / medium / heavy)
      - discount_x_weight: discount_offered * (weight_in_gms / WEIGHT_NORM)
      - calls_per_cost: customer_care_calls / cost_of_product
      - high_discount: 1 if discount_offered > 40 else 0
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Weight class bins
        df["weight_class"] = pd.cut(
            df["weight_in_gms"],
            bins=[-np.inf, 2500, 5000, np.inf],
            labels=["light", "medium", "heavy"],
        ).astype(str)

        # Interaction: discount scaled by normalized weight
        df["discount_x_weight"] = df["discount_offered"] * (
            df["weight_in_gms"] / WEIGHT_NORM
        )

        # Customer-care intensity per dollar (cost > 0 in this dataset; guard anyway)
        df["calls_per_cost"] = df["customer_care_calls"] / df["cost_of_product"].clip(lower=1)

        # High-discount flag
        df["high_discount"] = (df["discount_offered"] > 40).astype(int)

        return df


def build_preprocessor() -> Pipeline:
    """Construct the unfitted preprocessing pipeline.

    The pipeline is: FeatureEngineer -> ColumnTransformer(OneHot+passthrough).
    Output is a 2-D numpy array of floats.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("encode", column_transformer),
        ]
    )
    return pipeline


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Return the final feature names after fitting the pipeline."""
    encoder: ColumnTransformer = fitted_pipeline.named_steps["encode"]
    return list(encoder.get_feature_names_out())


def make_splits(
    df: pd.DataFrame,
    target_col: str = "delayed",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """70/15/15 stratified split. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off 15% test
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    # Then split remaining 85% into ~70/15 -> val is 15/85 ≈ 0.1765 of the remainder
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval,
        y_trval,
        test_size=0.15 / 0.85,
        stratify=y_trval,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Fit the preprocessor on the training split and save all artifacts.

    Outputs:
      - model/preprocessor.pkl                  (fitted Pipeline)
      - data/processed/X_{train,val,test}.npy   (transformed feature matrices)
      - data/processed/y_{train,val,test}.npy   (targets)
      - data/processed/feature_names.json       (final feature names)
      - data/processed/raw_{train,val,test}.csv (raw splits for EDA / debugging)
    """
    import json

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {RAW_CSV} ...")
    df = load_raw_data()
    print(f"  shape={df.shape}, target balance:\n{df['delayed'].value_counts(normalize=True).round(3)}")

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    pipeline = build_preprocessor()
    print("Fitting preprocessor on training data ...")
    Xt_train = pipeline.fit_transform(X_train)
    Xt_val = pipeline.transform(X_val)
    Xt_test = pipeline.transform(X_test)

    feat_names = get_feature_names(pipeline)
    print(f"  output shape (train) = {Xt_train.shape}, n_features = {len(feat_names)}")

    # Save processed arrays
    np.save(PROCESSED_DIR / "X_train.npy", Xt_train)
    np.save(PROCESSED_DIR / "X_val.npy", Xt_val)
    np.save(PROCESSED_DIR / "X_test.npy", Xt_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_val.npy", y_val.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    # Save raw splits too (used by EDA notebook and CLI test fixtures)
    X_train.assign(delayed=y_train).to_csv(PROCESSED_DIR / "raw_train.csv", index=False)
    X_val.assign(delayed=y_val).to_csv(PROCESSED_DIR / "raw_val.csv", index=False)
    X_test.assign(delayed=y_test).to_csv(PROCESSED_DIR / "raw_test.csv", index=False)

    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    joblib.dump(pipeline, MODEL_DIR / "preprocessor.pkl")
    print(f"\nSaved: {MODEL_DIR / 'preprocessor.pkl'}")
    print(f"Saved processed arrays + raw splits to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
