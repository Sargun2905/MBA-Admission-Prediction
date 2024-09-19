# MBA Admission Prediction Project

This project aims to predict MBA admission outcomes (`Admit`, `Deny`, `Waitlist`) based on various applicant features such as gender, GPA, GMAT score, work experience, and more. The dataset used is a synthetic dataset generated from Wharton Class of 2025's statistics.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)

---

## Project Overview

The main objective of this project is to build a machine learning model to predict the admission status of MBA applicants based on features such as:
- Gender
- GPA
- GMAT Score
- Work Experience
- Undergraduate Major
- Industry of Work Experience

### Goals:
- **Perform Exploratory Data Analysis (EDA)** to understand the dataset.
- **Build a classification model** to predict admission outcomes.
- **Evaluate the model** and identify areas of improvement, particularly focusing on the imbalance between the classes.

---

## Dataset

The dataset is stored in `MBA.csv` and contains the following columns:
- `application_id`: Unique identifier for each applicant.
- `gender`: Gender of the applicant (`Male`, `Female`).
- `international`: Whether the applicant is an international student (`True`, `False`).
- `gpa`: Grade Point Average of the applicant.
- `major`: Undergraduate major of the applicant (e.g., `STEM`, `Business`, `Humanities`).
- `race`: Racial background of the applicant.
- `gmat`: GMAT score of the applicant.
- `work_exp`: Number of years of work experience.
- `work_industry`: Industry of the applicant's previous work experience.
- `admission`: Admission status (`Admit`, `Deny`, `Waitlist`).

The dataset used in this project is the **MBA Admissions Dataset**, which was sourced from [Kaggle](https://www.kaggle.com/datasets/taweilo/mba-admission-dataset).

### Description
- The dataset contains synthetic data generated from the Wharton Class of 2025's statistics, including various attributes such as gender, GPA, GMAT score, work experience, and admission status.

---

## Exploratory Data Analysis (EDA)

In the EDA phase, we:
- Visualized the distribution of numerical features like `gpa`, `gmat`, and `work_exp`.
- Analyzed the correlation between `gpa`, `gmat`, and admission status.
- Handled missing values:
  - `race` column: Imputed with the most frequent value.
  - `admission` column: Rows with missing `admission` status were dropped.

---

## Data Preprocessing

- Categorical variables such as `gender`, `major`, `work_industry`, and `admission` were encoded using `LabelEncoder`.
- The dataset was split into training (80%) and testing (20%) sets.
- Target variable (`admission`) was label-encoded to map `Admit`, `Deny`, and `Waitlist` to integer values.

---

## Modeling

The **Random Forest Classifier** was selected as the primary model for this project. The classifier was trained on the training set, and predictions were made on the test set.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Label encoding the target variable (admission)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))
```
## Results

The **Random Forest Classifier** achieved an **accuracy of 0.82**. Below are the detailed performance metrics:

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Admit      | 0.46      | 0.22   | 0.30     | 196     |
| Deny       | 0.85      | 0.95   | 0.90     | 1025    |
| Waitlist   | 0.00      | 0.00   | 0.00     | 18      |

- **Accuracy**: 0.82
- **Weighted Average Precision**: 0.78
- **Weighted Average Recall**: 0.82
- **Weighted Average F1-Score**: 0.79

### Class Imbalance

The model performed well for predicting the `"Deny"` class but struggled with the `"Admit"` and `"Waitlist"` classes due to class imbalance.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Sargun2905/MBA-Admission-Prediction.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or Python scripts to execute the code.
4. Explore the results, evaluate the model, and experiment with improvements.

