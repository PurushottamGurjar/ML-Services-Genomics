import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils.downloader import download_file


def run_confusion(file_url):
    try:
        file_path = download_file(file_url)

        # ✅ Read RAW data (IMPORTANT)
        df = pd.read_csv(file_path)

        print("\n===== RAW DATA =====")
        print(df.head())
        print("\nColumns:", df.columns.tolist())

        # ✅ Basic check
        if df.shape[1] < 2:
            return {"error": "Dataset must have at least 2 columns (features + label)"}

        # ✅ Separate label BEFORE preprocessing
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

        print("\nLabel Column:", df.columns[-1])
        print("Sample Labels:", y.head())

        # ✅ Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna("unknown")

        # ✅ Encode categorical features
        X = pd.get_dummies(X)

        if X.shape[1] == 0:
            return {"error": "No usable features after preprocessing"}

        # ✅ Encode labels (handles string + numeric labels)
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

        # ✅ Check number of classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {
                "error": "Need at least 2 classes for classification",
                "classes_found": unique_classes.tolist(),
            }

        # ✅ Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # ✅ Train-test split (with stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ✅ Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # ✅ Predictions
        y_pred = model.predict(X_test)

        # ✅ Metrics
        cm = confusion_matrix(y_test, y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print("\nClass distribution:", np.unique(y, return_counts=True))

        return {
            "matrix": cm.tolist(),
            "labels": le.classes_.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    except Exception as e:
        return {"error": str(e)}