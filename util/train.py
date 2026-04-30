import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

def load_dataset():
    # Load both CSVs and concatenate them into one DataFrame
    attentive_df = pd.read_csv(os.path.join(PROCESSED_DIR, "attentive.csv"))
    distracted_df = pd.read_csv(os.path.join(PROCESSED_DIR, "distracted.csv"))
    df = pd.concat([attentive_df, distracted_df], ignore_index=True)

    # Shuffle the rows so classes are mixed (not all attentive first, then all distracted)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    return df

def train(df):
    # Split features (X) from labels (y)
    X = df.drop(columns=["label"])
    y = df["label"]

    # 80% training, 20% testing — random_state=42 makes the split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialise the Random Forest
    #   n_estimators=100    : number of decision trees in the forest
    #   class_weight=balanced: compensates if one class has more samples than the other
    #   random_state=42     : makes results reproducible
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    print("Training...")
    clf.fit(X_train, y_train)
    print("Done.\n")

    # Evaluate on the held-out test set
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["distracted", "attentive"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    return clf, X.columns.tolist()

def save_model(clf, feature_names):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the trained model to disk so it can be loaded by the main app later
    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(clf, model_path)
    print(f"\nModel saved -> {model_path}")

    # Save feature names so the live pipeline knows which columns to pass in
    feature_path = os.path.join(MODEL_DIR, "feature_names.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"Feature names saved -> {feature_path}")

if __name__ == "__main__":
    df = load_dataset()
    clf, feature_names = train(df)
    save_model(clf, feature_names)