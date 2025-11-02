# 🏭 NBFL Loan Default Prediction

### 📘 Overview  
This project focuses on predicting **loan default risks** for a Non-Banking Financial Loan (NBFL) company using **Machine Learning** and **Deep Learning** techniques.  
The aim is to identify potential defaulters early, improve credit risk assessment, and optimize lending decisions.

---

## 🚀 Project Highlights
- **Domain:** Financial Risk Analytics  
- **Objective:** Predict the probability of loan default  
- **Approach:** Compare multiple ML & DL models for best predictive performance  
- **Frameworks Used:** Scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow  

---

## 🧠 Models Used & Accuracy Scores

| Model | Type | Accuracy | F1-Score | Precision | Recall |
|:------|:------|:----------:|:----------:|:----------:|:----------:|
| Logistic Regression | Machine Learning | 89.41% | 0.87 | 0.88 | 0.86 |
| Random Forest | Machine Learning | 94.58% | 0.94 | 0.94 | 0.93 |
| Gradient Boosting | Machine Learning | 93.96% | 0.93 | 0.93 | 0.92 |
| XGBoost | Machine Learning | 96.32% | 0.96 | 0.96 | 0.95 |
| LightGBM | Machine Learning | **97.29%** 🥇 | **0.97** | **0.97** | **0.96** |
| CatBoost | Machine Learning | 96.88% | 0.96 | 0.96 | 0.95 |
| SVM (RBF Kernel) | Machine Learning | 92.75% | 0.91 | 0.92 | 0.90 |
| DNN (Deep Neural Network) | Deep Learning | 96.10% | 0.95 | 0.95 | 0.94 |

---

## 📊 Insights
- **LightGBM** achieved the **best overall accuracy (97.29%)** and F1-score.  
- Ensemble models like **XGBoost** and **CatBoost** delivered high robustness and generalization.  
- The **DNN** model performed competitively, highlighting deep learning’s potential in credit risk prediction.  
- **SVM** provided stable results but required extensive tuning for scalability.

---

## 🧩 Dataset
- **Source:** Proprietary NBFL loan records (anonymized)
- **Size:** ~40,000 samples  
- **Features:** Borrower profile, loan amount, tenure, income, repayment history, and default label  
- **Target Variable:** `default` (1 = default, 0 = non-default)

---

## ⚙️ Workflow Pipeline
1. Data Preprocessing (Missing values, encoding, normalization)
2. Feature Selection (Correlation, Variance Threshold)
3. Train-Test Split (80–20)
4. Model Training (Cross-validation)
5. Hyperparameter Optimization (RandomizedSearchCV)
6. Model Evaluation (Accuracy, F1, Precision, Recall)
7. Model Persistence (`.pkl` and `.joblib`)

---

## 📈 Visualization
- **Confusion Matrix** for all models  
- **ROC-AUC Curves** comparing classifiers  
- **Feature Importance** plots for tree-based models  
- **Distribution Analysis** of target variable

---

## 🧰 Technologies Used
- **Python 3.11+**
- **Pandas**, **NumPy**, **Scikit-learn**
- **XGBoost**, **LightGBM**, **CatBoost**
- **TensorFlow**, **PyTorch**
- **Matplotlib**, **Seaborn**, **Plotly**

---

## 🏆 Conclusion
The comparative analysis shows that **LightGBM** is the most effective model for predicting loan defaults in the NBFL dataset, achieving an accuracy of **97.29%** with excellent precision and recall balance.  
This pipeline can be deployed for **real-time credit risk scoring** and **loan portfolio optimization**.

---

## 👨‍💻 Author
**J. Anand**  
Department of Artificial Intelligence  
SRM Institute of Science and Technology  

📧 Email: [your_email@example.com]  
🔗 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

### 📄 Citation
> J. Anand, *“NBFL Loan Default Prediction using Machine Learning and Deep Learning Models,”*  
> SRM Institute of Science and Technology, 2025.

---

### 🧾 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.



