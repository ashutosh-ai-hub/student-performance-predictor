# 🎓 Student Performance Predictor

An AI-powered web application that predicts a student's final exam marks based on their study habits and academic history. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this project demonstrates an end-to-end Machine Learning workflow—from data processing and analysis to web deployment.

## 🚀 Key Features
- **Predictive Modeling:** Uses a Linear Regression algorithm to forecast grades with high precision.
- **Interactive UI:** A clean, responsive dashboard allowing users to input data via sliders and get instant results.
- **Real-Time Evaluation:** Displays live model metrics like **$R^2$ Score** and **Mean Absolute Error (MAE)**.
- **Data Visualizations:** Built-in correlation heatmaps and scatter plots to help understand feature importance and data trends.
- **Automated Logic:** Instant "Pass/Fail" status based on predicted results.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit

## 📊 Model Analysis
The model is trained on a dataset of **1,000 student records** focusing on 5 critical features:
1. **Study Hours**
2. **Attendance Percentage**
3. **Previous Exam Marks**
4. **Sleep Hours**
5. **Extracurricular Activities**

### Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Algorithm** | Linear Regression |
| **Test Split** | 20% |
| **$R^2$ Score** | **0.952** |
| **Avg. Error** | ~1.8 Marks |

## 📂 Project Structure
```plaintext
├── app.py              # Streamlit web application code
├── student_data.csv    # Dataset used for training/testing
├── notebook.ipynb      # Detailed EDA and Model Training process
├── requirements.txt    # Required Python libraries
└── image_a97b2f.png    # Application screenshot
