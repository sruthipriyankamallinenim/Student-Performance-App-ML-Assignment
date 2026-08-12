
# Student Performance Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to predict the GradeClass of students using demographic, academic, and extracurricular features. Five machine learning classification models were implemented and compared based on multiple evaluation metrics.

---

## 2. Dataset Description

The dataset contains 2,392 student records and 15 columns.

The target variable is:

- GradeClass

The following columns were removed:

- StudentID: It is only an identification variable.
- GPA: It was removed to avoid target leakage.

The final model uses 12 input features:

- Age
- Gender
- Ethnicity
- ParentalEducation
- StudyTimeWeekly
- Absences
- Tutoring
- ParentalSupport
- Extracurricular
- Sports
- Music
- Volunteering

The dataset contains no missing values.

---

## 3. GitHub Repository Link

GitHub Repository:

PASTE YOUR GITHUB LINK HERE

---

## 4. Models Used and Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7265 | 0.9109 | 0.7159 | 0.7265 | 0.7150 | 0.5877 |
| Decision Tree | 0.6743 | 0.8733 | 0.6799 | 0.6743 | 0.6722 | 0.5155 |
| KNN | 0.5741 | 0.7826 | 0.5399 | 0.5741 | 0.5492 | 0.3388 |
| Naive Bayes | 0.6722 | 0.8899 | 0.6647 | 0.6722 | 0.6527 | 0.5075 |
| Random Forest | 0.7182 | 0.9049 | 0.7250 | 0.7182 | 0.7110 | 0.5773 |

---

## 5. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved the best overall performance. It obtained the highest Accuracy, AUC, Recall, F1 Score, and MCC. |
| Decision Tree | Decision Tree achieved moderate performance and performed lower than Logistic Regression and Random Forest. |
| KNN | KNN showed the lowest performance among all five models. |
| Naive Bayes | Naive Bayes achieved moderate performance and was comparable to the Decision Tree. |
| Random Forest | Random Forest achieved strong performance and obtained the highest Precision. |

---

## 6. Overall Winner

### Logistic Regression

Logistic Regression was selected as the overall best model because it achieved the highest Accuracy, AUC, Recall, F1 Score, and MCC among the five models.

Although Random Forest achieved slightly higher Precision, Logistic Regression provided the best overall performance across the evaluation metrics.

## Conclusion

Based on the experimental results, Logistic Regression is the most suitable model among the five implemented models for predicting student GradeClass using the selected 12 input features.
