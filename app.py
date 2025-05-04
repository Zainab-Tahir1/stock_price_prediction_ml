# stonks_ml_final_working.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             silhouette_score, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib
from io import BytesIO

# ========================================
# Cyberpunk 2077 Theme
# ========================================

st.set_page_config(
    page_title="📈 Stonks ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #000000;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}

.stDataFrame {
    background-color: rgba(0, 25, 0, 0.9) !important;
    border: 1px solid #00ff00 !important;
    border-radius: 5px;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
}

.metric-card {
    background: rgba(0, 40, 0, 0.9) !important;
    border: 1px solid #00ff00;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
}

@keyframes neonPulse {
    0% { filter: drop-shadow(0 0 5px #00ff00); }
    50% { filter: drop-shadow(0 0 20px #00ff00); }
    100% { filter: drop-shadow(0 0 5px #00ff00); }
}

.neon-title {
    animation: neonPulse 2s infinite;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Core Functions with Validation
# ======================

def validate_and_preprocess(df):
    """Handle data validation and preprocessing with error checking"""
    try:
        # Clean column names
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        
        # Identify close price column
        close_aliases = ['close', 'closing_price', 'last_price', 'adj_close', 'price']
        for alias in close_aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: 'close'})

                break
        
        # Fallback to first numeric column if no close found
        if 'close' not in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in dataset")
            df = df.rename(columns={numeric_cols[0]: 'close'})
            st.warning(f"Using '{numeric_cols[0]}' as close price")
        
        # Convert to numeric and handle missing values
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['close'].ffill(inplace=True)
        df['close'].bfill(inplace=True)
        
        # Handle dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Data validation failed: {str(e)}")
        st.stop()

def handle_missing_data(df):
    """Interactive missing data handling with validation"""
    with st.expander("🔍 Missing Data Handling"):
        original_count = len(df)
        missing = df.isnull().sum()
        
        st.write("Missing values per column:")
        st.write(missing)
        
        method = st.radio("Data Imputation Protocol:", 
                         ["🚮 Purge NA", "📊 Fill with Mean", "📈 Fill with Median"])
        
        if st.button("Apply Imputation"):
            if method == "🚮 Purge NA":
                df = df.dropna()
            else:
                for col in df.select_dtypes(include=np.number):
                    if method == "📊 Fill with Mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].median())
            
            st.info(f"Removed {original_count - len(df)} rows")
            if len(df) == 0:
                st.error("All data removed during cleaning!")
                st.stop()
                
    return df

def detect_outliers(df):
    """Interactive outlier detection with safeguards"""
    with st.expander("📊 Anomaly Detection"):
        cols = st.multiselect("Select features for analysis:", 
                             df.select_dtypes(include=np.number).columns)
        
        original_count = len(df)
        if cols:
            for col in cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5*iqr
                upper_bound = q3 + 1.5*iqr
                
                if st.button(f"Purge anomalies in {col}", key=f"outlier_{col}"):
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
        if len(df) < original_count * 0.1:
            st.error("Over 90% data removed! Check outlier thresholds")
            st.stop()
            
    return df

# ======================
# Main Application Flow
# ======================

def main():
    st.markdown("""
    <div style="text-align: center;">
        <h1 class="neon-title">🚀 STONKS ML</h1>
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHprbG15eWY0bGdhOGU5MGViZ2Fyejh0d2ZnOTBqdnlxMW1jN2k5bCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/IgiVDEpoMTk0PEVbuW/giphy.gif" width="60%">
    </div>
    """, unsafe_allow_html=True)

    # Session State Management
    if 'pipeline_step' not in st.session_state:
        st.session_state.update({
            'pipeline_step': 0,
            'raw_df': None,
            'df': None,
            'model': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None
        })

    # Sidebar Configuration
    with st.sidebar:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnNscHdidTBzNW4wbXZrMnVrZGJ5amMzM2twYjdkd283MzduYmd3ZCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/RgzryV9nRCMHPVVXPV/giphy.gif")
        data_source = st.radio("Data Source:", ["📤 Upload", "🌐 Yahoo Finance"])
        model_type = st.selectbox("Analysis Type:", 
                                ["📈 Regression", "🔮 Classification", "🌀 Clustering"])
        
        st.markdown("---")
        st.markdown("**Pipeline Status:**")
        steps = ["Data Loaded", "Preprocessed", "Features Engineered", 
                "Data Split", "Model Trained", "Evaluated"]
        current_step = st.session_state.pipeline_step
        st.markdown("\n".join([f"{'🟢' if i < current_step else '⚫'} {step}" 
                             for i, step in enumerate(steps)]))

    # Data Loading
    if st.session_state.pipeline_step == 0:
        with st.container():
            st.markdown("## 📥 Phase 1: Data Acquisition")
            
            if data_source == "📤 Upload":
                uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
                if uploaded_file and st.button("🚀 Load Data"):
                    try:
                        df = pd.read_csv(uploaded_file)
                        df = validate_and_preprocess(df)
                        st.session_state.raw_df = df
                        st.session_state.pipeline_step = 1
                        st.success("Data loaded successfully!")
                        st.dataframe(df.head(3))
                    except Exception as e:
                        st.error(str(e))

            else:
                col1, col2 = st.columns(2)
                with col1:
                    ticker = st.text_input("Ticker Symbol", "AAPL").upper()
                with col2:
                    start = st.date_input("Start Date", datetime(2020,1,1))
                    end = st.date_input("End Date", datetime.today())
                
                if st.button("🌐 Fetch Market Data"):
                    with st.spinner("Accessing market data..."):
                        try:
                            df = yf.download(ticker, start=start, end=end)
                            if not df.empty:
                                df = df.reset_index().rename(columns={'Close': 'close'})
                                df = validate_and_preprocess(df)
                                st.session_state.raw_df = df
                                st.session_state.pipeline_step = 1
                                st.success("Data fetched successfully!")
                                st.dataframe(df.head(3))
                            else:
                                st.error("Invalid ticker or date range")
                        except Exception as e:
                            st.error(f"API Error: {str(e)}")

    # Data Preprocessing
    if st.session_state.pipeline_step == 1:
        with st.container():
            st.markdown("## 🧹 Phase 2: Data Preparation")
            
            if st.session_state.raw_df is not None:
                df = st.session_state.raw_df.copy()
                df = handle_missing_data(df)
                df = detect_outliers(df)
                
                if len(df) < 10:
                    st.error("Insufficient data after preprocessing (min 10 samples required)")
                    st.session_state.pipeline_step = 0
                    return
                
                if st.button("⚡ Finalize Preprocessing"):
                    st.session_state.df = df
                    st.session_state.pipeline_step = 2
                    st.success(f"Preprocessing complete! {len(df)} samples remaining")
                    st.dataframe(df.describe())

    # Feature Engineering
    if st.session_state.pipeline_step == 2:
        with st.container():
            st.markdown("## 🔧 Phase 3: Feature Engineering")
            
            df = st.session_state.df.copy()
            try:
                # Feature calculations
                df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
                df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
                delta = df['close'].diff().fillna(0)
                gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                df['rsi'] = 100 - (100 / (1 + rs))
                
                if model_type != "🌀 Clustering":
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    df = df.dropna()
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if model_type == "🔮 Classification":
                    numeric_cols.remove('target')
                elif model_type == "📈 Regression":
                    numeric_cols.remove('close')
                
                selected_features = st.multiselect("Select Features:", 
                                                  numeric_cols,
                                                  default=numeric_cols)
                
                if len(selected_features) == 0:
                    st.error("Select at least one feature!")
                    return
                
                if st.button("⚙️ Finalize Features"):
                    keep_cols = selected_features + ['target'] if model_type == "🔮 Classification" else selected_features + ['close']
                    st.session_state.df = df[keep_cols]
                    st.session_state.pipeline_step = 3
                    st.success(f"Selected {len(selected_features)} features")
                    
            except Exception as e:
                st.error(f"Feature engineering error: {str(e)}")
                st.session_state.pipeline_step = 1

    # Data Splitting
    if st.session_state.pipeline_step == 3:
        with st.container():
            st.markdown("## ✂️ Phase 4: Data Partitioning")
            
            df = st.session_state.df.copy()
            try:
                if model_type != "🌀 Clustering":
                    target = 'target' if model_type == "🔮 Classification" else 'close'
                    test_size = st.slider("Test Size (%)", 10, 40, 20)
                    
                    if len(df) < 20:
                        st.warning("Small dataset - consider reducing test size")
                        test_size = min(test_size, 30)
                    
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    min_test_samples = max(1, int(len(df)*0.05))
                    if (len(df)*test_size/100) < min_test_samples:
                        test_size = max(test_size, int(min_test_samples/len(df)*100))
                        st.warning(f"Adjusted test size to {test_size}% for minimum {min_test_samples} samples")
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size/100, 
                        random_state=42,
                        shuffle=True
                    )
                    
                    fig = px.pie(names=['Train', 'Test'], 
                                values=[len(X_train), len(X_test)],
                                color_discrete_sequence=['#00ff00', '#009900'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    X_train = df
                    X_test = df
                    y_train = pd.Series()
                    y_test = pd.Series()
                
                if st.button("🔀 Finalize Split"):
                    st.session_state.update({
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'pipeline_step': 4
                    })
                    st.success("Data split completed!")
                    
            except Exception as e:
                st.error(f"Data splitting failed: {str(e)}")
                st.session_state.pipeline_step = 2

    # Model Training
    if st.session_state.pipeline_step == 4:
        with st.container():
            st.markdown("## 🤖 Phase 5: Model Training")
            
            try:
                model_options = {
                    "📈 Regression": ["Linear", "Ridge", "Lasso"],
                    "🔮 Classification": ["Logistic", "SVM", "Random Forest"],
                    "🌀 Clustering": ["K-Means", "DBSCAN", "Hierarchical"]
                }
                
                model_choice = st.selectbox("Select Model", model_options[model_type])
                
                model = None
                if model_type == "📈 Regression":
                    if model_choice == "Linear":
                        model = make_pipeline(StandardScaler(), LinearRegression())
                    elif model_choice == "Ridge":
                        model = make_pipeline(StandardScaler(), Ridge())
                    else:
                        model = make_pipeline(StandardScaler(), Lasso())
                        
                elif model_type == "🔮 Classification":
                    if model_choice == "Logistic":
                        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                    elif model_choice == "SVM":
                        model = make_pipeline(StandardScaler(), SVC())
                    else:
                        model = make_pipeline(StandardScaler(), RandomForestClassifier())
                        
                else:
                    if model_choice == "K-Means":
                        n_clusters = st.slider("Clusters", 2, 5, 3)
                        model = make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters))
                    elif model_choice == "DBSCAN":
                        model = make_pipeline(StandardScaler(), DBSCAN())
                    else:
                        model = make_pipeline(StandardScaler(), AgglomerativeClustering())
                
                if st.button("🧠 Train Model"):
                    with st.spinner("Training in progress..."):
                        if model_type != "🌀 Clustering":
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                        else:
                            model.fit(st.session_state.X_train)
                            
                        st.session_state.model = model
                        st.session_state.pipeline_step = 5
                        st.success("Model trained successfully!")
                        
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.session_state.pipeline_step = 3

    # Model Evaluation
    if st.session_state.pipeline_step == 5:
        with st.container():
            st.markdown("## 📊 Phase 6: Model Evaluation")
            
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                if model_type == "📈 Regression":
                    preds = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
                        st.metric("R² Score", f"{r2_score(y_test, preds):.2f}")
                    with col2:
                        fig = px.scatter(x=y_test, y=preds, 
                                        labels={'x': 'Actual', 'y': 'Predicted'},
                                        trendline="lowess")
                        st.plotly_chart(fig, use_container_width=True)
                        
                elif model_type == "🔮 Classification":
                    preds = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
                        st.code(classification_report(y_test, preds))
                    with col2:
                        fig = px.imshow(confusion_matrix(y_test, preds),
                                      color_continuous_scale='Greens',
                                      labels=dict(x="Predicted", y="Actual"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    score = silhouette_score(X_test, model.predict(X_test))
                    st.metric("Silhouette Score", f"{score:.2f}")
                    pca = PCA(n_components=3)
                    components = pca.fit_transform(X_test)
                    fig = px.scatter_3d(components, color=model.predict(X_test),
                                      color_continuous_scale='Greens')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.pipeline_step = 6
                
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                st.session_state.pipeline_step = 4

    # Results Export
    if st.session_state.pipeline_step == 6:
        with st.container():
            st.markdown("## 📤 Phase 7: Results Export")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Predictions"):
                    try:
                        results = pd.DataFrame({
                            'Actual': st.session_state.y_test,
                            'Predicted': st.session_state.model.predict(st.session_state.X_test)
                        })
                        csv = results.to_csv(index=False)
                        st.download_button("Download CSV", csv, "predictions.csv")
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
                        
            with col2:
                if st.button("💾 Save Model"):
                    try:
                        with BytesIO() as buffer:
                            joblib.dump(st.session_state.model, buffer)
                            st.download_button("Download Model", 
                                             buffer.getvalue(), 
                                             "trained_model.pkl")
                    except Exception as e:
                        st.error(f"Model save failed: {str(e)}")
            
            st.markdown("---")
            st.image("https://media.giphy.com/media/3o7aD2d7hy9ktXNDP2/giphy.gif", width=300)
            st.success("Analysis Complete! 🎉")

if __name__ == "__main__":
    main()