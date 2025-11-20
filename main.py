import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import joblib

from pathlib import Path
from typing import Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint
from mlflow.models.signature import infer_signature

# Use a SQLite database for MLflow tracking instead of the default './mlruns' folder.
# This avoids deprecation warnings and stores all experiments in 'mlflow.db'.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

df = pd.read_csv("./data/pet_adoption_data.csv")

# -------------------------------------------------------
# Build Preprocessing Transformer
# -------------------------------------------------------

def build_preprocessor():
    numeric_features = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'),
             ['PetType', 'Breed', 'Color']),
            ('ordinal', OrdinalEncoder(categories=[['Small', 'Medium', 'Large']]),
             ['Size']),
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


# -------------------------------------------------------
# MLflow Experiment Setup
# -------------------------------------------------------

mlflow.set_experiment("AdoptionLikelihood_Project")

# -------------------------------------------------------
# NESTED RUN STRUCTURE
# -------------------------------------------------------

with mlflow.start_run(run_name="Master_AdoptionLikelihood_Experiment"):

    # ===========================================================
    # 1. BASELINE MODEL (Nested Run)
    # ===========================================================
    with mlflow.start_run(run_name="Baseline_Model", nested=True):

        df_clean = df.drop(columns=["PetID"], errors="ignore")
        X = df_clean.drop(columns=["AdoptionLikelihood"])
        y = df_clean["AdoptionLikelihood"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        baseline_model = DummyClassifier(strategy="most_frequent")
        baseline_model.fit(X_train, y_train)

        baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))

        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_metric("accuracy", baseline_acc)

        mlflow.sklearn.log_model(baseline_model, "baseline_model")

        print("[Baseline] Accuracy:", baseline_acc)

    # ===========================================================
    # 2. PREPROCESSING ONLY (Nested Run)
    # ===========================================================

    with mlflow.start_run(run_name="Preprocessing_Stage", nested=True):

        # Build and fit the preprocessor
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        # Create input example and infer signature
        input_example = X_train.head(1)
        signature = infer_signature(input_example, preprocessor.transform(input_example))

        # Log the preprocessor as a model with signature and input example
        mlflow.sklearn.log_model(
            preprocessor,
            artifact_path="preprocessing_pipeline",
            signature=signature,
            input_example=input_example
        )

        print("[Preprocessing] Transformer fitted and logged with signature and input example.")

    # ===========================================================
    # 3. FINAL MODEL TRAINING (Nested Run)
    # ===========================================================
    with mlflow.start_run(run_name="Final_Model", nested=True):

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        final_acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", final_acc)
        mlflow.sklearn.log_model(pipeline, "final_pipeline")

        # classification report as JSON artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        Path("artifacts").mkdir(exist_ok=True)

        with open("artifacts/classification_report.json", "w") as f:
            json.dump(report, f)

        with open("artifacts/confusion_matrix.json", "w") as f:
            json.dump(cm, f)

        mlflow.log_artifact("artifacts/classification_report.json")
        mlflow.log_artifact("artifacts/confusion_matrix.json")

        print("[Final Model] Accuracy:", final_acc)

    # ===========================================================
    # 4. HYPERPARAMETER SEARCH (Nested Run)
    # ===========================================================
    with mlflow.start_run(run_name="Hyperparameter_Search", nested=True):

        search_pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        param_dist = {
            'classifier__n_estimators': randint(100, 300),
            'classifier__max_depth': randint(5, 20),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 5),
            'classifier__max_features': ['sqrt', 'log2']
        }

        random_search = RandomizedSearchCV(
            search_pipeline,
            param_distributions=param_dist,
            n_iter=15,
            cv=5,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_score = random_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", best_score)

        mlflow.sklearn.log_model(random_search.best_estimator_, "best_model")

        print("[Hyperparameter Search] Best CV Score:", best_score)

print("All MLflow runs completed successfully.")
mlflow.end_run()