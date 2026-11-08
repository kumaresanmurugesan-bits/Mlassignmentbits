import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report, roc_curve
)

st.set_page_config(
    page_title="Netflix Content Classifier - ML Assignment 2",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E50914;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #E50914;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111111;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Netflix Content Classifier (Movie vs. TV Show)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">BITS Pilani M.Tech AIML — Machine Learning Assignment 2 | Student ID: <b>2026aj05717</b></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

selected_model_name = st.sidebar.selectbox("Select ML Classification Model", list(model_options.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Feature definitions
expected_features = [
    'release_year', 'year_added', 'month_added', 
    'title_length', 'title_word_count', 
    'desc_length', 'desc_word_count', 
    'cast_count', 'country_count', 'has_director',
    'is_us', 'is_india', 'is_uk',
    'is_drama', 'is_comedy', 'is_action', 'is_doc', 'is_international'
]
target_col = 'target'

# Load data
df_test = None
if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded CSV loaded successfully ({len(df_test)} rows)")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
else:
    default_test_path = "test_data.csv"
    if os.path.exists(default_test_path):
        df_test = pd.read_csv(default_test_path)
        st.sidebar.info(f"Using default test_data.csv ({len(df_test)} rows)")
    else:
        st.warning("Please upload a test dataset CSV to begin.")

if df_test is not None:
    # Verify columns
    missing_feats = [col for col in expected_features if col not in df_test.columns]
    if missing_feats:
        st.error(f"Missing required feature columns in test CSV: {missing_feats}")
    elif target_col not in df_test.columns:
        st.error(f"Target column '{target_col}' not found in uploaded CSV.")
    else:
        X_eval = df_test[expected_features]
        y_eval = df_test[target_col]

        # Load Model
        model_path = model_options[selected_model_name]
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please verify your repository structure.")
        else:
            pipeline = joblib.load(model_path)
            
            # Predict
            y_pred = pipeline.predict(X_eval)
            try:
                y_prob = pipeline.predict_proba(X_eval)[:, 1]
            except Exception:
                y_prob = pipeline.decision_function(X_eval)

            # Compute Evaluation Metrics
            acc = accuracy_score(y_eval, y_pred)
            auc = roc_auc_score(y_eval, y_prob)
            prec = precision_score(y_eval, y_pred, zero_division=0)
            rec = recall_score(y_eval, y_pred, zero_division=0)
            f1 = f1_score(y_eval, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_eval, y_pred)

            st.subheader(f"📊 Evaluation Metrics: {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            metrics = [
                ("Accuracy", f"{acc:.4f}", col1),
                ("AUC Score", f"{auc:.4f}", col2),
                ("Precision", f"{prec:.4f}", col3),
                ("Recall", f"{rec:.4f}", col4),
                ("F1 Score", f"{f1:.4f}", col5),
                ("MCC Score", f"{mcc:.4f}", col6),
            ]
            
            for label, val, col in metrics:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{val}</div>
                        <div class="metric-lbl">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualizations & Detailed Diagnostics
            tab1, tab2, tab3 = st.tabs(["📌 Confusion Matrix & ROC", "📋 Classification Report", "🔍 Sample Predictions"])

            with tab1:
                row_c1, row_c2 = st.columns(2)
                with row_c1:
                    st.markdown("##### Confusion Matrix")
                    cm = confusion_matrix(y_eval, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                                xticklabels=['TV Show (0)', 'Movie (1)'],
                                yticklabels=['TV Show (0)', 'Movie (1)'])
                    ax_cm.set_xlabel('Predicted Label')
                    ax_cm.set_ylabel('Actual Label')
                    st.pyplot(fig_cm)
                
                with row_c2:
                    st.markdown("##### Receiver Operating Characteristic (ROC)")
                    fpr, tpr, _ = roc_curve(y_eval, y_prob)
                    fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.8))
                    ax_roc.plot(fpr, tpr, color='#E50914', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
                    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.legend(loc="lower right")
                    ax_roc.grid(True, alpha=0.3)
                    st.pyplot(fig_roc)

            with tab2:
                st.markdown("##### Classification Report")
                report_dict = classification_report(y_eval, y_pred, target_names=['TV Show (0)', 'Movie (1)'], output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4))

            with tab3:
                st.markdown("##### Sample Test Data Predictions")
                preview_df = X_eval.head(15).copy()
                preview_df['Actual_Target'] = y_eval.head(15).values
                preview_df['Predicted_Target'] = y_pred[:15]
                preview_df['Probability(Movie)'] = np.round(y_prob[:15], 4)
                st.dataframe(preview_df)

st.markdown("---")
st.caption("BITS Pilani WILP M.Tech AIML — Machine Learning Assignment 2")
