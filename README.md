# InsightFlow — AI-Powered Data Analytics Platform

InsightFlow is a Python-based data analytics platform that combines data preprocessing, exploratory data analysis, visualization, and machine learning in an interactive Streamlit application.

It helps users transform raw datasets into meaningful analytical insights through data quality assessment, automated cleaning, statistical analysis, predictive modeling, and recommendations.

## Features

- **Data Upload:** Load datasets through CSV file uploads.
- **Data Profiling:** Examine dataset dimensions, missing values, duplicates, and column types.
- **Data Cleaning:** Handle missing values and remove duplicate records.
- **Exploratory Data Analysis:** Generate descriptive statistics, KPIs, and correlation analysis.
- **Data Visualization:** Explore distributions, relationships, and correlations through charts.
- **Machine Learning:**
  - Classification using Logistic Regression, Random Forest, and Gradient Boosting.
  - Regression using Linear Regression.
  - Customer or record segmentation using K-Means clustering.
  - Anomaly detection using Isolation Forest.
- **Insights and Recommendations:** Generate analytical findings and recommendations from the processed data.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Application Framework | Streamlit |
| Testing | Pytest |

## Project Structure

```text
InsightFlow/
├── app.py
├── data/
│   ├── sample_data.csv
│   └── messy_data.csv
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── analytics.py
│   ├── visualization.py
│   ├── ml_models.py
│   ├── insights.py
│   ├── recommendations.py
│   └── logger.py
├── tests/
├── logs/
├── output/
├── frontend/
├── .gitignore
└── LICENSE
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gaytrij/insightflow-data-analytics.git
cd insightflow-data-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn openpyxl pytest
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can access the application in your browser.

## Running Tests

Run the test suite using:

```bash
python -m pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Gayatri Kishor Jadhav**

GitHub: [@gaytrij](https://github.com/gaytrij)