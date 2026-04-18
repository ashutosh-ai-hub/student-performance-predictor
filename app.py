import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.write("Developed by **Ashutosh Srivastava**")
st.title("🎓 Student Performance Predictor")
st.markdown("Predict final exam marks using study habits and academic history.")
st.divider()

# ── FIX 1: Relative path — no more C:\\Users\\DELL\\... ───
@st.cache_data
def load_and_train():
    df = pd.read_csv(r"student_data.csv")   # relative path
    X = df[['study_hours', 'attendance', 'previous_marks', 'sleep_hours', 'extracurricular']]
    y = df['final_marks']
    # FIX 2: Proper train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, df, r2, mae

model, df, r2, mae = load_and_train()

# ── Layout: two columns ───────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Enter Student Details")

    study_hours     = st.slider("📚 Study Hours per Day",  min_value=1,   max_value=10,  value=6,  step=1)
    attendance      = st.slider("🏫 Attendance (%)",        min_value=30,  max_value=100, value=75, step=1)
    previous_marks  = st.slider("📝 Previous Marks",        min_value=25,  max_value=95,  value=65, step=1)
    sleep_hours     = st.slider("😴 Sleep Hours per Night", min_value=4,   max_value=9,   value=7,  step=1)
    extracurricular = st.selectbox("🏅 Extracurricular Activities", options=[0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")

    predict_btn = st.button("Predict Final Marks", use_container_width=True, type="primary")

with col_result:
    st.subheader("Prediction Result")

    if predict_btn:
        input_data  = np.array([[study_hours, attendance, previous_marks, sleep_hours, extracurricular]])
        prediction  = model.predict(input_data)[0]
        status      = "Pass ✅" if prediction >= 40 else "Fail ❌"
        status_color = "green" if prediction >= 40 else "red"

        st.metric(label="Predicted Final Marks", value=f"{prediction:.1f} / 100")
        st.markdown(f"**Status:** :{status_color}[{status}]")

        # Progress bar
        st.progress(int(min(prediction, 100)))

        # Grade band
        if prediction >= 75:
            grade = "A — Distinction"
        elif prediction >= 60:
            grade = "B — First Class"
        elif prediction >= 50:
            grade = "C — Second Class"
        elif prediction >= 40:
            grade = "D — Pass"
        else:
            grade = "F — Fail"
        st.info(f"Grade Band: **{grade}**")

        # Feature importance mini chart
        st.markdown("**What influenced this prediction:**")
        features    = ['study_hours', 'attendance', 'previous_marks', 'sleep_hours', 'extracurricular']
        values      = [study_hours, attendance, previous_marks, sleep_hours, extracurricular]
        coefs       = model.coef_
        contributions = np.array(coefs) * np.array(values)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors = ['steelblue' if c > 0 else 'tomato' for c in contributions]
        ax.barh(features, contributions, color=colors)
        ax.set_xlabel('Contribution to Score')
        ax.set_title('Feature Contribution')
        ax.axvline(0, color='gray', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Adjust the sliders on the left and click **Predict Final Marks**.")

# ── Data Overview section ─────────────────────────────────
st.divider()
st.subheader("📊 Dataset Overview & Model Performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset Size",      f"{len(df):,} records")
m2.metric("Model R² Score",    f"{r2:.3f}",  help="Evaluated on 20% test set — not training data")
m3.metric("Mean Abs. Error",   f"{mae:.1f} marks")
m4.metric("Features Used",     "5")

# Correlation heatmap + scatter in two columns
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df.drop(columns='final_marks').join(df['final_marks']).corr(),
                annot=True, fmt='.2f', cmap='Blues', ax=ax, linewidths=0.4)
    ax.set_title('Feature Correlations')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with c2:
    st.markdown("**Study Hours vs Final Marks**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df['study_hours'], df['final_marks'], alpha=0.4, color='steelblue', edgecolors='white', linewidth=0.3)
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Final Marks')
    ax.set_title('Study Hours vs Final Marks')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Raw data preview
with st.expander("View Raw Dataset"):
    st.dataframe(df.head(20), use_container_width=True)
