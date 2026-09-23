import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.insights import generate_insights

from src.recommendations import (
    generate_recommendations
)

from src.data_cleaner import (
    get_data_quality_report,
    clean_dataset
)

from src.analytics import (
    generate_summary_statistics,
    generate_correlation_matrix,
    generate_kpis
)

from src.visualization import (
    plot_numeric_distribution,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_scatter
)

from src.ml_models import (
    train_classification_model,
    compare_classification_models,
    train_regression_model,
    perform_clustering,
    detect_anomalies
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Data Analytics Platform",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 AI-Powered Data Processing & Analytics Platform"
)

st.write(
    """
    Upload a CSV dataset to automatically perform
    data validation, cleaning, analytics,
    visualization, automated insights, machine learning,
    model comparison, anomaly detection, and
    actionable recommendations.
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "Dataset uploaded successfully!"
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        st.stop()


    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.header(
        "📋 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(
            df.isnull().sum().sum()
        )
    )

    col4.metric(
        "Duplicate Rows",
        int(
            df.duplicated().sum()
        )
    )


    # ========================================================
    # RAW DATASET
    # ========================================================

    with st.expander(
        "🔍 View Raw Dataset"
    ):

        st.dataframe(
            df,
            width="stretch"
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.header(
        "🔎 Data Quality Analysis"
    )

    quality_report = get_data_quality_report(
        df
    )

    st.dataframe(
        quality_report,
        width="stretch"
    )


    # ========================================================
    # DATA CLEANING
    # ========================================================

    st.header(
        "🧹 Data Cleaning"
    )

    cleaned_df = clean_dataset(
        df
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows After Cleaning",
        cleaned_df.shape[0]
    )

    col2.metric(
        "Missing Values After Cleaning",
        int(
            cleaned_df.isnull().sum().sum()
        )
    )

    col3.metric(
        "Duplicate Rows After Cleaning",
        int(
            cleaned_df.duplicated().sum()
        )
    )


    with st.expander(
        "📄 View Cleaned Dataset"
    ):

        st.dataframe(
            cleaned_df,
            width="stretch"
        )


    # ========================================================
    # DOWNLOAD CLEANED DATASET
    # ========================================================

    cleaned_csv = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key="cleaned_dataset_download"
    )


    # ========================================================
    # ANALYTICS & KPIs
    # ========================================================

    st.header(
        "📈 Analytics & KPIs"
    )

    kpis = generate_kpis(
        cleaned_df
    )

    kpi_columns = st.columns(
        min(len(kpis), 4)
    )

    for index, (
        key,
        value
    ) in enumerate(
        kpis.items()
    ):

        kpi_columns[
            index % len(kpi_columns)
        ].metric(
            key,
            value
        )


    # ========================================================
    # AUTOMATED INSIGHTS
    # ========================================================

    st.header(
        "🤖 Automated Data Insights"
    )

    insights = generate_insights(
        cleaned_df
    )

    for insight in insights:

        st.info(
            insight
        )


    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    st.subheader(
        "📊 Summary Statistics"
    )

    summary = generate_summary_statistics(
        cleaned_df
    )

    if not summary.empty:

        st.dataframe(
            summary,
            width="stretch"
        )

    else:

        st.info(
            "No numerical columns available."
        )


    # ========================================================
    # ANALYTICS REPORT
    # ========================================================

    st.subheader(
        "📥 Export Analytics Report"
    )

    report_buffer = io.StringIO()

    report_buffer.write(
        "AI-Powered Data Analytics Report\n"
    )

    report_buffer.write(
        "=" * 50 + "\n\n"
    )

    report_buffer.write(
        "DATASET OVERVIEW\n"
    )

    report_buffer.write(
        "-" * 30 + "\n"
    )

    report_buffer.write(
        f"Rows: {cleaned_df.shape[0]}\n"
    )

    report_buffer.write(
        f"Columns: {cleaned_df.shape[1]}\n"
    )

    report_buffer.write(
        f"Missing Values: "
        f"{int(cleaned_df.isnull().sum().sum())}\n"
    )

    report_buffer.write(
        f"Duplicate Rows: "
        f"{int(cleaned_df.duplicated().sum())}\n\n"
    )


    report_buffer.write(
        "KEY PERFORMANCE INDICATORS\n"
    )

    report_buffer.write(
        "-" * 30 + "\n"
    )

    for key, value in kpis.items():

        report_buffer.write(
            f"{key}: {value}\n"
        )


    report_buffer.write(
        "\nSUMMARY STATISTICS\n"
    )

    report_buffer.write(
        "-" * 30 + "\n"
    )

    if not summary.empty:

        report_buffer.write(
            summary.to_csv()
        )


    report_buffer.write(
        "\nAUTOMATED INSIGHTS\n"
    )

    report_buffer.write(
        "-" * 30 + "\n"
    )

    for insight in insights:

        report_buffer.write(
            f"- {insight}\n"
        )


    report_data = (
        report_buffer
        .getvalue()
        .encode("utf-8")
    )


    st.download_button(
        label="📥 Download Analytics Report",
        data=report_data,
        file_name="analytics_report.csv",
        mime="text/csv",
        key="analytics_report_download"
    )


    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    st.subheader(
        "🔗 Correlation Analysis"
    )

    correlation = generate_correlation_matrix(
        cleaned_df
    )

    if not correlation.empty:

        st.dataframe(
            correlation,
            width="stretch"
        )

        fig = plot_correlation_heatmap(
            cleaned_df
        )

        st.pyplot(
            fig
        )

        plt.close(fig)

    else:

        st.info(
            "Correlation analysis requires numerical columns."
        )


    # ========================================================
    # DATA VISUALIZATION
    # ========================================================

    st.header(
        "📊 Data Visualization"
    )

    numerical_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = cleaned_df.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()


    if numerical_columns:

        st.subheader(
            "📉 Numerical Distribution"
        )

        selected_column = st.selectbox(
            "Select a numerical column",
            numerical_columns,
            key="distribution_column"
        )

        fig = plot_numeric_distribution(
            cleaned_df,
            selected_column
        )

        st.pyplot(
            fig
        )

        plt.close(fig)


    if categorical_columns:

        st.subheader(
            "📊 Categorical Distribution"
        )

        selected_category = st.selectbox(
            "Select a categorical column",
            categorical_columns,
            key="categorical_column"
        )

        fig = plot_categorical_distribution(
            cleaned_df,
            selected_category
        )

        st.pyplot(
            fig
        )

        plt.close(fig)


    if len(numerical_columns) >= 2:

        st.subheader(
            "🔬 Relationship Between Variables"
        )

        col1, col2 = st.columns(2)

        with col1:

            x_column = st.selectbox(
                "Select X-axis",
                numerical_columns,
                key="scatter_x"
            )

        with col2:

            y_column = st.selectbox(
                "Select Y-axis",
                numerical_columns,
                key="scatter_y"
            )

        fig = plot_scatter(
            cleaned_df,
            x_column,
            y_column
        )

        st.pyplot(
            fig
        )

        plt.close(fig)


    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    st.header(
        "🤖 Machine Learning"
    )

    model_results_for_recommendations = None
    feature_importance_for_recommendations = None
    anomaly_results_for_recommendations = None


    if len(numerical_columns) >= 2:

        target_column = st.selectbox(
            "Select target column",
            numerical_columns,
            key="ml_target"
        )

        target_unique_values = cleaned_df[
            target_column
        ].nunique()

        st.caption(
            f"Selected target: {target_column} | "
            f"Unique values: {target_unique_values}"
        )


        ml_task = st.selectbox(
            "Select ML task",
            [
                "Classification",
                "Regression",
                "Clustering"
            ],
            key="ml_task"
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if ml_task == "Classification":

            if target_unique_values < 2:

                st.warning(
                    "Classification requires at least 2 classes."
                )

            elif target_unique_values > 10:

                st.warning(
                    "This target has more than 10 unique values."
                )

            else:

                st.subheader(
                    "🧠 Model Comparison"
                )

                if st.button(
                    "🚀 Compare Classification Models",
                    key="compare_models_button"
                ):

                    try:

                        (
                            comparison_df,
                            trained_models,
                            predictions_dict,
                            feature_importance,
                            y_test,
                            feature_names
                        ) = compare_classification_models(
                            cleaned_df,
                            target_column
                        )


                        model_results_for_recommendations = (
                            comparison_df
                        )

                        feature_importance_for_recommendations = (
                            feature_importance
                        )


                        st.subheader(
                            "📊 Model Performance Comparison"
                        )

                        display_df = (
                            comparison_df.copy()
                        )

                        metric_columns = [
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "F1 Score"
                        ]

                        if "ROC-AUC" in display_df.columns:

                            metric_columns.append(
                                "ROC-AUC"
                            )

                        for column in metric_columns:

                            if column in display_df.columns:

                                display_df[
                                    column
                                ] = (
                                    display_df[
                                        column
                                    ].round(4)
                                )


                        st.dataframe(
                            display_df,
                            width="stretch"
                        )


                        st.subheader(
                            "📈 Model Performance"
                        )

                        chart_df = (
                            comparison_df
                            .set_index("Model")
                            [metric_columns]
                        )

                        st.bar_chart(
                            chart_df
                        )


                        st.subheader(
                            "🔍 Feature Importance"
                        )

                        available_models = list(
                            feature_importance.keys()
                        )

                        if available_models:

                            selected_importance_model = (
                                st.selectbox(
                                    "Select model",
                                    available_models,
                                    key="feature_importance_model"
                                )
                            )

                            importance_df = (
                                feature_importance[
                                    selected_importance_model
                                ].copy()
                            )

                            st.dataframe(
                                importance_df,
                                width="stretch"
                            )

                            importance_plot = (
                                importance_df
                                .head(15)
                                .sort_values(
                                    "Importance"
                                )
                            )

                            fig, ax = plt.subplots(
                                figsize=(8, 5)
                            )

                            ax.barh(
                                importance_plot[
                                    "Feature"
                                ],
                                importance_plot[
                                    "Importance"
                                ]
                            )

                            ax.set_xlabel(
                                "Importance"
                            )

                            ax.set_ylabel(
                                "Feature"
                            )

                            ax.set_title(
                                "Feature Importance"
                            )

                            plt.tight_layout()

                            st.pyplot(
                                fig
                            )

                            plt.close(fig)


                        st.subheader(
                            "📊 Confusion Matrix"
                        )

                        selected_confusion_model = (
                            st.selectbox(
                                "Select model for confusion matrix",
                                list(
                                    trained_models.keys()
                                ),
                                key="confusion_model"
                            )
                        )

                        selected_predictions = (
                            predictions_dict[
                                selected_confusion_model
                            ]
                        )

                        cm = confusion_matrix(
                            y_test,
                            selected_predictions
                        )

                        fig, ax = plt.subplots(
                            figsize=(6, 5)
                        )

                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm
                        )

                        disp.plot(
                            ax=ax,
                            cmap="Blues"
                        )

                        ax.set_title(
                            f"Confusion Matrix - "
                            f"{selected_confusion_model}"
                        )

                        st.pyplot(
                            fig
                        )

                        plt.close(fig)


                        model_results_csv = (
                            comparison_df
                            .to_csv(index=False)
                            .encode("utf-8")
                        )

                        st.download_button(
                            label="📥 Download Model Comparison",
                            data=model_results_csv,
                            file_name="model_comparison.csv",
                            mime="text/csv",
                            key="model_comparison_download"
                        )


                    except Exception as e:

                        st.error(
                            f"Model comparison failed: {e}"
                        )


        # ====================================================
        # REGRESSION
        # ====================================================

        elif ml_task == "Regression":

            if target_unique_values < 2:

                st.warning(
                    "Regression requires a target with "
                    "more than one unique value."
                )

            else:

                if st.button(
                    "▶ Run Regression",
                    key="regression_button"
                ):

                    try:

                        (
                            model,
                            metrics,
                            y_test,
                            predictions
                        ) = train_regression_model(
                            cleaned_df,
                            target_column
                        )


                        st.subheader(
                            "Regression Results"
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric(
                            "MAE",
                            f"{metrics['MAE']:.2f}"
                        )

                        c2.metric(
                            "MSE",
                            f"{metrics['MSE']:.2f}"
                        )

                        c3.metric(
                            "RMSE",
                            f"{metrics['RMSE']:.2f}"
                        )

                        c4.metric(
                            "R² Score",
                            f"{metrics['R2']:.2f}"
                        )


                        st.subheader(
                            "📈 Actual vs Predicted"
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )

                        ax.scatter(
                            y_test,
                            predictions,
                            alpha=0.7
                        )

                        min_value = min(
                            y_test.min(),
                            predictions.min()
                        )

                        max_value = max(
                            y_test.max(),
                            predictions.max()
                        )

                        ax.plot(
                            [
                                min_value,
                                max_value
                            ],
                            [
                                min_value,
                                max_value
                            ],
                            linestyle="--"
                        )

                        ax.set_xlabel(
                            "Actual Values"
                        )

                        ax.set_ylabel(
                            "Predicted Values"
                        )

                        ax.set_title(
                            "Actual vs Predicted Values"
                        )

                        plt.tight_layout()

                        st.pyplot(
                            fig
                        )

                        plt.close(fig)


                    except Exception as e:

                        st.error(
                            f"Regression failed: {e}"
                        )


        # ====================================================
        # CLUSTERING
        # ====================================================

        else:

            number_of_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=6,
                value=3,
                key="cluster_number"
            )


            if st.button(
                "▶ Run Clustering",
                key="clustering_button"
            ):

                try:

                    (
                        model,
                        clustered_df,
                        scaled_data,
                        cluster_features
                    ) = perform_clustering(
                        cleaned_df,
                        n_clusters=number_of_clusters
                    )


                    st.subheader(
                        "Clustering Results"
                    )

                    st.dataframe(
                        clustered_df,
                        width="stretch"
                    )


                    if len(cluster_features) >= 2:

                        st.subheader(
                            "🔵 Cluster Visualization"
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 6)
                        )

                        ax.scatter(
                            scaled_data[:, 0],
                            scaled_data[:, 1],
                            c=clustered_df[
                                "Cluster"
                            ],
                            alpha=0.7
                        )

                        ax.set_xlabel(
                            cluster_features[0]
                        )

                        ax.set_ylabel(
                            cluster_features[1]
                        )

                        ax.set_title(
                            "K-Means Cluster Visualization"
                        )

                        plt.tight_layout()

                        st.pyplot(
                            fig
                        )

                        plt.close(fig)


                except Exception as e:

                    st.error(
                        f"Clustering failed: {e}"
                    )


    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    st.header(
        "🚨 Anomaly Detection"
    )

    st.write(
        """
        Identify unusual records automatically using
        Isolation Forest.
        """
    )


    if len(numerical_columns) >= 2:

        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="anomaly_contamination"
        )


        if st.button(
            "🚨 Detect Anomalies",
            key="anomaly_detection_button"
        ):

            try:

                (
                    anomaly_model,
                    anomaly_df,
                    anomaly_scaled_data,
                    anomaly_features
                ) = detect_anomalies(
                    cleaned_df,
                    contamination=contamination
                )


                anomaly_results_for_recommendations = (
                    anomaly_df
                )


                total_records = len(
                    anomaly_df
                )

                anomaly_count = int(
                    (
                        anomaly_df[
                            "Anomaly"
                        ] == -1
                    ).sum()
                )

                normal_count = int(
                    (
                        anomaly_df[
                            "Anomaly"
                        ] == 1
                    ).sum()
                )

                anomaly_percentage = (
                    anomaly_count /
                    total_records *
                    100
                )


                st.subheader(
                    "📊 Anomaly Summary"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Total Records",
                    total_records
                )

                c2.metric(
                    "Normal Records",
                    normal_count
                )

                c3.metric(
                    "Anomalies",
                    anomaly_count
                )

                c4.metric(
                    "Anomaly %",
                    f"{anomaly_percentage:.2f}%"
                )


                st.subheader(
                    "🔎 Detected Records"
                )

                display_anomalies = anomaly_df[
                    anomaly_df[
                        "Anomaly"
                    ] == -1
                ].copy()

                if not display_anomalies.empty:

                    st.dataframe(
                        display_anomalies,
                        width="stretch"
                    )

                else:

                    st.success(
                        "No anomalies were detected."
                    )


                st.subheader(
                    "📈 Normal vs Anomaly"
                )

                distribution = (
                    anomaly_df[
                        "Anomaly Label"
                    ]
                    .value_counts()
                )

                st.bar_chart(
                    distribution
                )


                if len(anomaly_features) >= 2:

                    st.subheader(
                        "🔬 Anomaly Visualization"
                    )

                    fig, ax = plt.subplots(
                        figsize=(8, 6)
                    )

                    normal_mask = (
                        anomaly_df[
                            "Anomaly"
                        ] == 1
                    )

                    anomaly_mask = (
                        anomaly_df[
                            "Anomaly"
                        ] == -1
                    )


                    ax.scatter(
                        anomaly_scaled_data[
                            normal_mask,
                            0
                        ],
                        anomaly_scaled_data[
                            normal_mask,
                            1
                        ],
                        alpha=0.6,
                        label="Normal"
                    )


                    ax.scatter(
                        anomaly_scaled_data[
                            anomaly_mask,
                            0
                        ],
                        anomaly_scaled_data[
                            anomaly_mask,
                            1
                        ],
                        alpha=0.8,
                        marker="x",
                        s=80,
                        label="Anomaly"
                    )


                    ax.set_xlabel(
                        anomaly_features[0]
                    )

                    ax.set_ylabel(
                        anomaly_features[1]
                    )

                    ax.set_title(
                        "Isolation Forest Anomaly Detection"
                    )

                    ax.legend()

                    plt.tight_layout()

                    st.pyplot(
                        fig
                    )

                    plt.close(fig)


                st.subheader(
                    "📐 Anomaly Scores"
                )

                score_df = anomaly_df[
                    [
                        "Anomaly Label",
                        "Anomaly Score"
                    ]
                ].copy()

                score_df = score_df.sort_values(
                    "Anomaly Score"
                )

                st.dataframe(
                    score_df.head(20),
                    width="stretch"
                )


                anomaly_csv = (
                    anomaly_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label="📥 Download Anomaly Report",
                    data=anomaly_csv,
                    file_name="anomaly_detection_report.csv",
                    mime="text/csv",
                    key="anomaly_report_download"
                )


            except Exception as e:

                st.error(
                    f"Anomaly detection failed: {e}"
                )


    # ========================================================
    # AUTOMATED RECOMMENDATIONS
    # ========================================================

    st.header(
        "💡 Automated Recommendations"
    )

    st.write(
        """
        Recommendations are generated from the dataset's
        quality characteristics, statistical relationships,
        model results, and anomaly-detection results.
        """
    )


    recommendations = generate_recommendations(
        cleaned_df,
        anomaly_df=anomaly_results_for_recommendations,
        model_results=model_results_for_recommendations,
        feature_importance=feature_importance_for_recommendations
    )


    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.success(
            f"**Recommendation {index}:** "
            f"{recommendation}"
        )


    # ========================================================
    # DOWNLOAD RECOMMENDATIONS
    # ========================================================

    recommendation_text = (
        "AI-Powered Data Analytics "
        "Recommendations\n"
    )

    recommendation_text += (
        "=" * 55 +
        "\n\n"
    )


    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        recommendation_text += (
            f"{index}. "
            f"{recommendation}\n\n"
        )


    st.download_button(
        label="📥 Download Recommendations",
        data=recommendation_text.encode(
            "utf-8"
        ),
        file_name="automated_recommendations.txt",
        mime="text/plain",
        key="recommendations_download"
    )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        "📂 Upload a CSV file to begin analysis."
    )