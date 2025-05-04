# 📈 Interactive Financial ML App using Streamlit

**Course**: Programming for Finance (AF3005)  
**Instructor**: [Instructor's Name]  
**Submission Date**: 4 May 2025  
**Student**: [Your Full Name]

## 🎯 Objective

This project demonstrates an interactive **machine learning application** developed with **Streamlit** that allows users to upload Kragle datasets or fetch real-time stock data from Yahoo Finance. The app guides users through a **step-by-step ML workflow**, from data loading to model evaluation and visualization.

---

## 🚀 Key Features

- Upload financial data (CSV) or fetch via **Yahoo Finance API**
- Step-by-step ML pipeline with buttons & progress notifications
- Real-time feedback using `st.success`, `st.warning`, `st.info`
- Choose one ML model:  
  - Linear Regression  
  - Logistic Regression  
  - K-Means Clustering
- Interactive data visualizations using **Plotly**
- GIFs and themed UI for better user engagement
- Downloadable results and dynamic feature handling

---

## 📊 Machine Learning Pipeline

| Step              | Description                                | Visual Output                      |
|-------------------|--------------------------------------------|------------------------------------|
| Load Data         | Upload or fetch financial data             | Table preview + success message    |
| Preprocessing     | Handle missing values, outliers            | Summary stats + notification       |
| Feature Engineering | Transform/Select features               | Feature importance/chart           |
| Train/Test Split  | Split dataset                              | Pie chart of train/test ratio      |
| Model Training    | Train ML model                             | Completion notification            |
| Evaluation        | Display metrics and results                | Accuracy, confusion matrix, etc.   |
| Results           | Visualize predictions or clusters          | Plotly graphs, scatterplots        |

---

## 🛠️ Technologies Used

- **Python Libraries**:  
  `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `plotly`, `yfinance`


## 📥 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/FinancialML_Streamlit_App.git
   cd FinancialML_Streamlit_App
