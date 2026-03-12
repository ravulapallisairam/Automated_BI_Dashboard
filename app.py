import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(page_title="Automated BI Dashboard", layout="wide")

# STYLE
st.markdown("""
<style>
.stApp {
background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
color:white;
}
[data-testid="stMetricValue"]{
font-size:28px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("# 🚀 Automated AI Business Intelligence Dashboard")

# SIDEBAR
st.sidebar.title("Dashboard Menu")

menu = st.sidebar.radio(
    "Navigate",
    ["Overview","Charts","Correlation","AI Insights","Prediction","Data"]
)

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # LOAD DATA
    df = pd.read_csv(uploaded_file)

    # ---------------- DATA CLEANING ----------------
    df = df.drop_duplicates()

    # Remove unwanted columns like Unnamed:0
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Reset index
    df = df.reset_index(drop=True)

    # Convert date columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # FILTERS
    st.sidebar.subheader("Filters")

    if categorical_cols:

        filter_col = st.sidebar.selectbox("Filter Column", categorical_cols)

        values = st.sidebar.multiselect(
            "Select Values",
            sorted(df[filter_col].dropna().unique())
        )

        if values:
            df = df[df[filter_col].isin(values)]

    # OVERVIEW
    if menu == "Overview":

        st.subheader("Dataset Overview")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Rows",df.shape[0])
        col2.metric("Columns",df.shape[1])
        col3.metric("Missing Values",df.isnull().sum().sum())
        col4.metric("Numeric Features",len(numeric_cols))

        st.dataframe(df.head())

        st.subheader("Missing Value Analysis")

        missing = df.isnull().sum()
        missing_df = missing[missing > 0]

        st.dataframe(missing_df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Clean Dataset",
            csv,
            "clean_dataset.csv",
            "text/csv"
        )

    # CHARTS
    elif menu == "Charts":

        st.subheader("Automatic Data Visualization")

        if numeric_cols:

            selected_num = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig = px.histogram(df,x=selected_num)

            st.plotly_chart(fig,use_container_width=True)

        if categorical_cols:

            selected_cat = st.selectbox(
                "Select Categorical Column",
                categorical_cols
            )

            top = df[selected_cat].value_counts().nlargest(10).reset_index()
            top.columns=[selected_cat,"Count"]

            fig = px.bar(top,x=selected_cat,y="Count")
            st.plotly_chart(fig,use_container_width=True)

    # CORRELATION
    elif menu == "Correlation":

        if len(numeric_cols)>1:

            st.subheader("Correlation Heatmap")

            corr=df[numeric_cols].corr()

            fig=px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r"
            )

            st.plotly_chart(fig,use_container_width=True)

            fig2,ax=plt.subplots(figsize=(8,5))
            sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)
            st.pyplot(fig2)

    # AI INSIGHTS
    elif menu == "AI Insights":

        st.subheader("AI Automated Insights")

        if numeric_cols:

            highest_col = df[numeric_cols].sum().idxmax()
            lowest_col = df[numeric_cols].sum().idxmin()

            st.success(f"Highest overall metric: {highest_col}")
            st.warning(f"Lowest overall metric: {lowest_col}")

            max_value = df[numeric_cols].max().max()
            st.info(f"Maximum value in dataset: {max_value}")

    # PREDICTION
    elif menu == "Prediction":

        st.subheader("Machine Learning Prediction")

        if len(numeric_cols)>1:

            target = st.selectbox("Target Column",numeric_cols)

            X = df[numeric_cols].drop(columns=[target])
            y = df[target]

            X_train,X_test,y_train,y_test = train_test_split(
                X,y,test_size=0.2
            )

            model = LinearRegression()
            model.fit(X_train,y_train)

            score=model.score(X_test,y_test)

            st.success(f"Model Accuracy: {round(score*100,2)}%")

    # DATA
    elif menu == "Data":

        st.subheader("Full Dataset")

        st.dataframe(df)

        col1,col2,col3 = st.columns(3)

        col1.write("Numeric Columns")
        col1.write(numeric_cols)

        col2.write("Categorical Columns")
        col2.write(categorical_cols)

        col3.write("Date Columns")
        col3.write(date_cols)

else:
    st.info("Please upload a CSV dataset to start analysis.")

st.markdown("---")
st.markdown("AI Powered Automated BI Dashboard")