import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecast App",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00C9A7;
    }
    .stButton>button {
        background-color: #00C9A7;
        color: black;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
# HEADER WITH LOGO
col1, col2 = st.columns([1, 4])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.title("📊 Sales Forecasting Dashboard")
    st.markdown("### Predict future sales with AI 🚀")

# ---------------- SIDEBAR ----------------
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("## Sales Forecasting App")
st.sidebar.markdown("Built with ❤️ using AI")
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
periods = st.sidebar.slider("📅 Days to Predict", 30, 365, 90)

# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Detect date column
    possible_date_cols = ['Date', 'Order Date', 'date', 'Invoice Date']
    date_col = next((col for col in possible_date_cols if col in df.columns), None)

    if date_col is None:
        st.error("❌ No valid Date column found")
    elif 'Sales' not in df.columns:
        st.error("❌ No Sales column found")
    else:
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Sales", f"{int(df['Sales'].sum())}")
        col2.metric("Average Sales", f"{int(df['Sales'].mean())}")
        col3.metric("Records", len(df))

        # ---------------- DATA PREVIEW ----------------
        with st.expander("📁 View Raw Data"):
            st.write(df.head())

        # ---------------- PLOTLY SALES TREND ----------------
        st.subheader("📈 Interactive Sales Trend")

        fig_trend = px.line(
            df,
            x='Date',
            y='Sales',
            title='Sales Over Time'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ---------------- PROPHET MODEL ----------------
        df_prophet = df[['Date', 'Sales']]
        df_prophet.columns = ['ds', 'y']

        with st.spinner("⏳ Training model..."):
            model = Prophet()
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

        st.success("✅ Forecast Generated Successfully!")

        # ---------------- PLOTLY FORECAST ----------------
        st.subheader("📊 Interactive Forecast")

        fig_forecast = go.Figure()

        # Actual data
        fig_forecast.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Sales'],
            mode='lines',
            name='Actual'
        ))

        # Predicted data
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast'
        ))

        fig_forecast.update_layout(title="Actual vs Forecast")

        st.plotly_chart(fig_forecast, use_container_width=True)

        # ---------------- COMPONENTS ----------------
        st.subheader("📉 Trend & Seasonality (Static)")

        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)

        # ---------------- DOWNLOAD ----------------
        st.subheader("📥 Download Forecast")

        csv = forecast[['ds', 'yhat']].to_csv(index=False)

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv"
        )
        st.markdown("---")
st.markdown("© 2026 Sales Forecasting App | Built by You 🚀")