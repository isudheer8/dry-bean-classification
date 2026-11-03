# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify dry bean samples into their corresponding varieties using geometrical measurements extracted from bean images.

The system uses five supervised machine learning classifiers:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models are trained and evaluated using the same dataset and held-out test split. Their performance is compared using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit application is provided so that a user can upload a labelled test CSV, evaluate an individual saved model, or compare all five saved models without retraining them.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset page:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
**DOI:** https://doi.org/10.24432/C50S4B

The dataset contains **13,611 instances**, **16 numerical input features**, and one target column named `Class`. The target represents one of seven dry bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

The 16 input features describe the geometrical properties of each bean, including area, perimeter, axis lengths, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, and shape factors.

### Dataset split and preprocessing

- Training split: 80% (10,888 records)
- Testing split: 20% (2,723 records)
- Random state: `42`
- Stratification: performed using the target class to preserve class proportions.
- Feature scaling: `StandardScaler` is applied through a Pipeline for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM.
- Decision Tree: trained without feature scaling.
- The held-out test data is not used to fit the models.

The supplied `test_data.csv` contains the 16 input features and the actual `Class` target column. It is used by the Streamlit application to calculate evaluation metrics.

## 3. GitHub Repository Link

**GitHub Repository:**  
[Add your GitHub repository URL here after creating the repository.]

The repository should contain the source code, trained model files, test dataset, results, dependency file, and this README.

### Project structure

```text
dry-bean-classification/
│
├── dataset/
│   └── dry_bean_dataset.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knearest_neighbors.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
├── results/
│   └── model_comparison.csv
│
├── screenshots/
│   └── bits_virtual_lab_execution.png
│
├── app.py
├── train_models.py
├── download_dataset.py
├── test_data.csv
├── requirements.txt
└── README.md
```

The `screenshots/` folder should contain the genuine screenshot captured during execution in the BITS Virtual Lab. Do not use a locally captured screenshot as a substitute for the required BITS Virtual Lab evidence.

## 4. Models Used and Performance Comparison

The five classifiers were evaluated on the same held-out test set. The following are the results obtained during the training and evaluation run.

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9448 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| K-Nearest Neighbors | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | **0.9221** | **0.9949** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

**Metric averaging:** Precision, Recall, and F1 Score are weighted averages for the multiclass problem. AUC is macro-averaged using the one-vs-rest approach. MCC is calculated as a multiclass correlation coefficient.

### Observations for each model

**1. Logistic Regression**

- Achieved 92.07% accuracy and a 0.9209 weighted F1 Score.
- Its AUC score of 0.9948 indicates strong class-separation performance.
- It performed close to the SVM model while using a relatively simple linear classification approach.

**2. Decision Tree Classifier**

- Achieved 89.17% accuracy, the lowest accuracy among the five evaluated models.
- Its AUC score was 0.9448, also the lowest among the compared models.
- The result suggests that this Decision Tree configuration generalised less effectively than the other evaluated classifiers on the held-out test set.

**3. K-Nearest Neighbors**

- Achieved 91.66% accuracy and a 0.9168 weighted F1 Score.
- Its AUC score was 0.9833.
- It provided competitive performance, although it remained slightly below Logistic Regression and SVM on the reported metrics.

**4. Gaussian Naive Bayes**

- Achieved 89.79% accuracy and a 0.8981 weighted F1 Score.
- Its AUC score was 0.9916, which was relatively high despite its lower accuracy.
- This indicates that its probability-based class ranking was strong, while its final class predictions were less accurate than those of Logistic Regression and SVM.

**5. Support Vector Machine**

- Achieved the highest accuracy of 92.21%.
- It also obtained the highest AUC (0.9949), weighted Precision (0.9224), Recall (0.9221), F1 Score (0.9222), and MCC (0.9059).
- Its results were marginally higher than Logistic Regression across the reported metrics.

### Overall Best Model

**Support Vector Machine (SVM)** was the best-performing model in this experiment based on the reported held-out test results. It achieved 92.21% accuracy and a weighted F1 Score of 0.9222.

Logistic Regression was a close second, with 92.07% accuracy and a weighted F1 Score of 0.9209. The small difference between these two models should be considered when selecting a model for deployment.

## 5. Streamlit Application

**Live Streamlit App:**  
[Add your deployed Streamlit Community Cloud URL here.]

### Application features

- Upload a labelled test dataset in CSV format.
- Validate the required 16 input features and the `Class` target column.
- Select one of the five saved classifiers for individual evaluation.
- Evaluate all five classifiers on the same uploaded test dataset.
- Display Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.
- Display a confusion matrix and classification report for an individually selected model.
- Download prediction results and the all-model comparison table.

The application loads the trained Joblib model files from the `model/` directory. It does not retrain models when a test CSV is uploaded.

### Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/isudheer8/dry-bean-classification
   cd dry-bean-classification
   ```

2. Create and activate a Python virtual environment, or use an existing environment.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal, upload `test_data.csv`, and choose an evaluation mode.

### Deploy on Streamlit Community Cloud

1. Push the complete project to a GitHub repository.
2. Sign in to Streamlit Community Cloud using the GitHub account that has access to the repository.
3. Create a new app and select the project repository and branch.
4. Set `app.py` as the application entry point.
5. Deploy the application.
6. Open the deployed URL and test the CSV upload and model evaluation features.

Ensure that the five `.joblib` files and `requirements.txt` are committed to the repository. The deployed app needs these files to load the trained classifiers.

## 6. Execution Evidence

**BITS Virtual Lab screenshot:**  
Add the genuine screenshot showing the project execution in the BITS Virtual Lab to the submitted PDF and, if required, to the repository's `screenshots/` folder.

The screenshot must be captured from the actual BITS Virtual Lab environment. It should clearly show the project or its execution as required by the assignment.

## 7. Notes

- All five models use the same dataset and held-out test split for the reported comparison.
- The Streamlit application evaluates saved models; it does not train models on uploaded test data.
- The uploaded evaluation CSV must contain all 16 expected feature columns and the target column `Class`.
- For reproducible Joblib model loading, use a compatible scikit-learn version in the training and deployment environments.
