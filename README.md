# Netflix Content Classification (Movie vs. TV Show)
**BITS Pilani — Work Integrated Learning Programmes Division**  
**M.Tech in Artificial Intelligence & Machine Learning (AIML)**  
**Course:** Machine Learning | **Assignment:** 2  
**Student Name:** kumaresan murugesan  
**Student ID:** 2026aj05717  

---

## 1. Problem Statement
The objective of this project is to build, evaluate, and deploy an end-to-end Machine Learning classification pipeline that classifies Netflix media titles into **Movie (1)** or **TV Show (0)** based on metadata attributes such as release timeline, description length, cast demographics, production origin, and genres. The solution demonstrates the complete ML lifecycle: data engineering, training five distinct classifiers, rigorous metric evaluation, interactive UI development using Streamlit, and cloud deployment.

---

## 2. Dataset Description
- **Dataset Choice:** Netflix Movies and TV Shows Dataset (Kaggle/Public Domain)
- **Minimum Instance Requirement:** > 500 (Provided: **8,807 instances**)
- **Minimum Feature Requirement:** > 12 (Provided: **18 engineered input features**)
- **Target Variable:** `target` (Binary: `1` for Movie, `0` for TV Show)
- **Dataset Composition:** 6,131 Movies (69.6%) and 2,676 TV Shows (30.4%)

### Extracted & Engineered Feature Attributes:
1. `release_year`: The original release year of the title.
2. `year_added`: Year when title was published to the Netflix streaming platform.
3. `month_added`: Numerical month when title was introduced.
4. `title_length`: Character length of the content title.
5. `title_word_count`: Total word count in the title string.
6. `desc_length`: Total character length of synopsis/plot description.
7. `desc_word_count`: Word count of synopsis/plot description.
8. `cast_count`: Number of credited main actors/actresses.
9. `country_count`: Number of contributing production countries.
10. `has_director`: Binary indicator (1 if director listed, 0 if NaN).
11. `is_us`: Binary flag indicating US production involvement.
12. `is_india`: Binary flag indicating Indian production involvement.
13. `is_uk`: Binary flag indicating UK production involvement.
14. `is_drama`: Binary flag indicating classification under 'Dramas'.
15. `is_comedy`: Binary flag indicating classification under 'Comedies'.
16. `is_action`: Binary flag indicating classification under 'Action & Adventure'.
17. `is_doc`: Binary flag indicating classification under 'Documentaries'.
18. `is_international`: Binary flag indicating classification under 'International'.

---

## 3. GitHub Repository Link
- **GitHub Repository:** `https://github.com/kumaresanmurugesan-bits/Mlassignmentbits/tree/main`  
- **Live Streamlit App:** `https://bitsmlassignmet-kkgxnljks7wvw4w3s3jrjy.streamlit.app/`

---

## 4. Models Used & Comparison Table
All five classification models were trained and evaluated on an identical stratified split (80% Train: 7,045 samples, 20% Test: 1,762 samples). 

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9523 | 0.9625 | 0.9576 | 0.9747 | 0.9661 | 0.8863 |
| **Decision Tree** | 0.9444 | 0.9452 | 0.9527 | 0.9682 | 0.9604 | 0.8674 |
| **KNN** | 0.9489 | 0.9573 | 0.9452 | 0.9837 | 0.9641 | 0.8780 |
| **Naive Bayes** | 0.9506 | 0.9577 | 0.9560 | 0.9739 | 0.9649 | 0.8822 |
| **Support Vector Machine** | **0.9546** | **0.9686** | **0.9570** | **0.9788** | **0.9678** | **0.8917** |

---

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Logistic Regression performed exceptionally well, achieving an accuracy of 95.23% and an AUC of 0.9625. The high recall (97.47%) and MCC of 0.8863 prove that the engineered metadata features possess strong linear separability with minimal multicollinearity penalties after standard scaling. |
| **Decision Tree** | With tree depth regularized to max_depth=6 to avoid overfitting, the Decision Tree achieved an accuracy of 94.44% and AUC of 0.9452. It successfully isolated high-information splits (e.g., `has_director` and `is_doc`), but had slightly lower recall and MCC (0.8674) compared to ensemble and kernel-based methods due to axis-aligned boundary constraints. |
| **KNN** | K-Nearest Neighbors (k=7 with Euclidean distance on standardized features) delivered the highest recall across all models (98.37%) and an accuracy of 94.89%. It was highly effective at grouping similar titles in feature space, though slight false positives slightly reduced its precision to 94.52%. |
| **Naive Bayes** | Gaussian Naive Bayes yielded a solid accuracy of 95.06% and AUC of 0.9577. Despite its strong independence assumption across correlated metadata features (such as title length and word count), it proved surprisingly robust and fast to compute, with an F1 score of 0.9649. |
| **Support Vector Machine** | SVM with an RBF kernel proved to be the top performer overall, achieving the highest Accuracy (95.46%), AUC (0.9686), F1-Score (0.9678), and MCC (0.8917). The radial basis function effectively mapped the complex interactions between continuous metadata and binary genre flags into high-dimensional space with maximum margin separation. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM with RBF Kernel)** is the undisputed winner. It attained superior metrics across almost every dimension (Accuracy: 95.46%, AUC: 0.9686, MCC: 0.8917), demonstrating outstanding generalization and discrimination power between Movies and TV Shows. |

---

## 6. Streamlit Web Application Features
1. **Interactive Test Data Upload:** Users can upload any test CSV containing the 18 engineered features and target.
2. **Dynamic Model Selection:** Switch seamlessly between all 5 trained classifiers stored in `/model`.
3. **Automated Metric Cards:** Instant display of Accuracy, AUC, Precision, Recall, F1, and MCC.
4. **Visual Diagnostics:** Includes an interactive Seaborn Confusion Matrix, Scikit-learn ROC curve with area under the curve, and full tabular Classification Report.
