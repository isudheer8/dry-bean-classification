import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

DATA_PATH = "dataset/dry_bean_dataset.csv"

MODEL_FOLDER = "model"
RESULT_FOLDER = "results"

TEST_RATIO = 0.20
SEED = 42

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

print("---------------------------------   DRY BEAN CLASSIFICATION SYSTEM   --------------------------------------")

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run download_dataset.py first."
    )

data = pd.read_csv(DATA_PATH)

target_column = "Class"

if target_column not in data.columns:
    raise ValueError("Target column 'Class' is missing.")

if data.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values.")

features = data.drop(columns=[target_column])
labels = data[target_column]

print("\nDataset Information")
print("-" * 40)
print("Total records :", len(data))
print("Input features:", features.shape[1])
print("Target classes:", labels.nunique())
print("Class labels  :", sorted(labels.unique()))

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=TEST_RATIO,
    random_state=SEED,
    stratify=labels
)

print("\nDataset Split")
print("-" * 40)
print("Training records:", len(X_train))
print("Testing records :", len(X_test))

classifiers = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=SEED
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=SEED
    ),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=SEED
        ))
    ])

}

evaluation_results = []

print("\n\nMODEL TRAINING AND EVALUATION")

for model_name, classifier in classifiers.items():

    print(f"\nTraining {model_name}...")
    
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    # Obtain probability estimates for AUC
    probabilities = classifier.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=classifier.classes_
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, predictions)

    evaluation_results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + ".joblib"
    )

    model_path = os.path.join(MODEL_FOLDER, filename)

    joblib.dump(classifier, model_path)

    print("-" * 45)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    print(f"Saved model: {model_path}")

comparison = pd.DataFrame(evaluation_results)

comparison_path = os.path.join(
    RESULT_FOLDER,
    "model_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\n\nFINAL MODEL COMPARISON")
print("=" * 100)

print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

test_dataset = X_test.copy()

test_dataset[target_column] = y_test

test_dataset.to_csv(
    "test_data.csv",
    index=False
)

print("\nTest dataset exported successfully.")
print("File: test_data.csv")

best_model = comparison.loc[
    comparison["F1 Score"].idxmax()
]

print("\nBEST PERFORMING MODEL")
print("-" * 45)

print("Model:", best_model["ML Model Name"])
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"F1 Score: {best_model['F1 Score']:.4f}")
print(f"MCC: {best_model['MCC']:.4f}")

print("       TRAINING COMPLETED SUCCESSFULLY")

print("\nGenerated Files:")

print("\nTrained Models:")
for filename in sorted(os.listdir(MODEL_FOLDER)):
    if filename.endswith(".joblib"):
        print(" -", filename)

print("\nEvaluation Results:")
print(" -", comparison_path)

print("\nTest Dataset:")
print(" - test_data.csv")

print("\nAll five classifiers have been trained and evaluated.")