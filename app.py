import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

MODEL_DIRECTORY = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    color: #374151 !important;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #374151 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #166534 !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifiers():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIRECTORY,
            filename
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        loaded_models[model_name] = joblib.load(
            model_path
        )

    return loaded_models


try:
    classifiers = load_classifiers()

except Exception as error:
    st.error(f"Unable to load trained models: {error}")
    st.stop()

st.title("🌱 Dry Bean Classification System")

st.markdown(
    """
    ### Machine Learning Based Multiclass Classification

    This application predicts the variety of dry beans using
    geometrical characteristics and evaluates five different
    machine learning classification algorithms.
    """
)

st.divider()

st.sidebar.title("Navigation")

selected_page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Model Evaluation",
        "Model Comparison"
    ]
)


if selected_page == "Project Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "13,611")
    col2.metric("Input Features", "16")
    col3.metric("Target Classes", "7")

    st.subheader("About the Dataset")

    st.write(
        """
        The Dry Bean Dataset contains geometrical measurements
        of seven different varieties of dry beans.

        The objective is to classify each bean into its
        corresponding variety using its physical characteristics.
        """
    )

    bean_classes = [
        "BARBUNYA",
        "BOMBAY",
        "CALI",
        "DERMASON",
        "HOROZ",
        "SEKER",
        "SIRA"
    ]

    st.subheader("Classification Categories")

    st.write(", ".join(bean_classes))

    st.subheader("Machine Learning Algorithms")

    for model_name in MODEL_FILES:

        st.write(f"✓ {model_name}")

    st.info(
        "Upload a labelled test CSV under Model Evaluation "
        "to calculate classification performance."
    )



elif selected_page == "Model Evaluation":

    st.header("Evaluate Classification Models")

    st.write(
        "Upload one labelled test CSV containing the 16 input features "
        "and the target column 'Class'. The same uploaded file can be used "
        "to evaluate one model or compare all five models."
    )

    # One shared uploader for both evaluation modes.
    uploaded_file = st.file_uploader(
        "Upload Test Dataset",
        type=["csv"],
        key="test_dataset_uploader"
    )

    if uploaded_file is not None:

        try:
            test_data = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")
            st.dataframe(
                test_data.head(),
                use_container_width=True
            )

            # Use the saved model to obtain the expected feature names.
            reference_model = classifiers["Logistic Regression"]
            expected_features = list(reference_model.feature_names_in_)

            missing_features = [
                feature for feature in expected_features
                if feature not in test_data.columns
            ]

            if missing_features:
                st.error(f"Missing input features: {missing_features}")
                st.stop()

            if "Class" not in test_data.columns:
                st.error("The target column 'Class' is missing.")
                st.stop()

            if test_data.empty:
                st.error("The uploaded dataset is empty.")
                st.stop()

            X_test = test_data[expected_features].copy()
            y_test = test_data["Class"].copy()

            if X_test.isnull().any().any() or y_test.isnull().any():
                st.error("The uploaded dataset contains missing values.")
                st.stop()

            try:
                X_test = X_test.astype(float)
            except (ValueError, TypeError):
                st.error("All 16 input features must contain numeric values.")
                st.stop()

            if not np.isfinite(X_test.to_numpy()).all():
                st.error("The input features contain infinite or non-finite values.")
                st.stop()

            valid_classes = set(reference_model.classes_)
            unknown_classes = set(y_test.unique()) - valid_classes

            if unknown_classes:
                st.error(
                    f"The target column contains unknown classes: "
                    f"{sorted(unknown_classes)}"
                )
                st.stop()

            st.success(
                f"Test dataset loaded successfully: {len(test_data):,} records."
            )

            st.divider()

            evaluation_mode = st.radio(
                "Choose Evaluation Mode",
                [
                    "Evaluate Individual Model",
                    "Evaluate All Models"
                ],
                horizontal=True,
                key="evaluation_mode"
            )

            def calculate_metrics(model, features, actual):
                """Calculate the six required metrics for one saved model."""

                predicted = model.predict(features)
                probabilities = model.predict_proba(features)

                accuracy = accuracy_score(actual, predicted)
                precision = precision_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0
                )
                recall = recall_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0
                )
                f1 = f1_score(
                    actual,
                    predicted,
                    average="weighted",
                    zero_division=0
                )
                mcc = matthews_corrcoef(actual, predicted)

                # Macro one-vs-rest AUC requires every model class
                # to be present in the uploaded test target.
                model_classes = list(model.classes_)
                if set(model_classes).issubset(set(actual.unique())):
                    auc = roc_auc_score(
                        actual,
                        probabilities,
                        labels=model_classes,
                        multi_class="ovr",
                        average="macro"
                    )
                else:
                    auc = np.nan

                return {
                    "Accuracy": accuracy,
                    "AUC": auc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "MCC": mcc,
                    "Predictions": predicted
                }

            # --------------------------------------------------
            # Evaluate one selected model
            # --------------------------------------------------
            if evaluation_mode == "Evaluate Individual Model":

                selected_model_name = st.selectbox(
                    "Select Classification Algorithm",
                    list(classifiers.keys()),
                    key="individual_model_selection"
                )

                if st.button(
                    "Evaluate Selected Model",
                    type="primary",
                    key="evaluate_individual_model"
                ):
                    model = classifiers[selected_model_name]

                    results = calculate_metrics(
                        model,
                        X_test,
                        y_test
                    )

                    st.subheader(f"Results: {selected_model_name}")

                    metric_row_one = st.columns(3)
                    metric_row_one[0].metric(
                        "Accuracy", f"{results['Accuracy']:.4f}"
                    )
                    metric_row_one[1].metric(
                        "AUC Score",
                        f"{results['AUC']:.4f}"
                        if not np.isnan(results["AUC"]) else "N/A"
                    )
                    metric_row_one[2].metric(
                        "Precision", f"{results['Precision']:.4f}"
                    )

                    metric_row_two = st.columns(3)
                    metric_row_two[0].metric(
                        "Recall", f"{results['Recall']:.4f}"
                    )
                    metric_row_two[1].metric(
                        "F1 Score", f"{results['F1 Score']:.4f}"
                    )
                    metric_row_two[2].metric(
                        "MCC", f"{results['MCC']:.4f}"
                    )

                    predictions = results["Predictions"]

                    prediction_table = pd.DataFrame({
                        "Actual Variety": y_test.to_numpy(),
                        "Predicted Variety": predictions,
                        "Prediction Status": [
                            "Correct" if actual == predicted else "Incorrect"
                            for actual, predicted in zip(y_test, predictions)
                        ]
                    })

                    st.divider()
                    st.subheader("Bean Classification Results")
                    st.dataframe(
                        prediction_table,
                        use_container_width=True
                    )

                    class_names = list(model.classes_)
                    matrix = confusion_matrix(
                        y_test,
                        predictions,
                        labels=class_names
                    )

                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 7))
                    sns.heatmap(
                        matrix,
                        annot=True,
                        fmt="d",
                        cmap="Greens",
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=ax
                    )
                    ax.set_xlabel("Predicted Variety")
                    ax.set_ylabel("Actual Variety")
                    ax.set_title(f"{selected_model_name} Confusion Matrix")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader("Classification Report")
                    report = classification_report(
                        y_test,
                        predictions,
                        labels=class_names,
                        output_dict=True,
                        zero_division=0
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True
                    )

                    download_table = test_data.copy()
                    download_table["Predicted_Class"] = predictions
                    download_table["Prediction_Status"] = prediction_table[
                        "Prediction Status"
                    ]

                    st.download_button(
                        "Download Prediction Results",
                        data=download_table.to_csv(index=False).encode("utf-8"),
                        file_name="bean_predictions.csv",
                        mime="text/csv",
                        key="download_individual_predictions"
                    )

            # --------------------------------------------------
            # Evaluate and compare all five saved models
            # --------------------------------------------------
            else:

                st.info(
                    "All five saved classifiers will be evaluated against "
                    "the same uploaded test dataset. No model is retrained."
                )

                if st.button(
                    "Evaluate All Models",
                    type="primary",
                    key="evaluate_all_models"
                ):
                    comparison_rows = []

                    progress = st.progress(0)
                    model_items = list(classifiers.items())

                    for index, (model_name, model) in enumerate(model_items):
                        with st.spinner(f"Evaluating {model_name}..."):
                            model_results = calculate_metrics(
                                model,
                                X_test,
                                y_test
                            )

                        comparison_rows.append({
                            "ML Model Name": model_name,
                            "Accuracy": model_results["Accuracy"],
                            "AUC": model_results["AUC"],
                            "Precision": model_results["Precision"],
                            "Recall": model_results["Recall"],
                            "F1 Score": model_results["F1 Score"],
                            "MCC": model_results["MCC"]
                        })

                        progress.progress((index + 1) / len(model_items))

                    comparison_df = pd.DataFrame(comparison_rows)

                    st.success("All five models evaluated successfully.")

                    st.subheader("Comparative Performance Analysis")
                    st.dataframe(
                        comparison_df.style.format({
                            "Accuracy": "{:.4f}",
                            "AUC": "{:.4f}",
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "F1 Score": "{:.4f}",
                            "MCC": "{:.4f}"
                        }, na_rep="N/A"),
                        use_container_width=True
                    )

                    winner = comparison_df.loc[
                        comparison_df["F1 Score"].idxmax()
                    ]

                    st.success(
                        f"Best model on this uploaded test dataset: "
                        f"{winner['ML Model Name']} "
                        f"(F1 Score: {winner['F1 Score']:.4f})"
                    )

                    st.subheader("Accuracy Comparison")
                    accuracy_chart = comparison_df.set_index(
                        "ML Model Name"
                    )["Accuracy"]
                    st.bar_chart(accuracy_chart)

                    st.download_button(
                        "Download All-Model Comparison",
                        data=comparison_df.to_csv(index=False).encode("utf-8"),
                        file_name="all_models_comparison.csv",
                        mime="text/csv",
                        key="download_all_models_comparison"
                    )

        except Exception as error:
            st.error(f"An error occurred during evaluation: {error}")

elif selected_page == "Model Comparison":

    st.header("Classification Model Comparison")

    results_path = os.path.join(
        "results",
        "model_comparison.csv"
    )

    if os.path.exists(results_path):

        comparison = pd.read_csv(results_path)

        st.write(
            "Performance comparison using the same "
            "held-out test dataset."
        )

        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.subheader("Accuracy Comparison")

        chart_data = comparison.set_index(
            "ML Model Name"
        )["Accuracy"]

        st.bar_chart(chart_data)

        winner = comparison.loc[
            comparison["F1 Score"].idxmax()
        ]

        st.success(
            f"Best performing model based on F1 Score: "
            f"{winner['ML Model Name']} "
            f"({winner['F1 Score']:.4f})"
        )

    else:

        st.warning(
            "Model comparison results not found. "
            "Run train_models.py first."
        )

st.divider()

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)
