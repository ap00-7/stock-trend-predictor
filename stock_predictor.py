import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Stock Price Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    /* Main background */
    [data-testid="stMainBlockContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
        padding: 0 !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px 15px;
        border-left: 3px solid #58a6ff;
    }
    
    /* Chart styling */
    .plotly-container {
        background-color: #0e1117 !important;
    }
    
    /* Headers - Responsive */
    h1 {
        color: #58a6ff !important;
        font-size: clamp(24px, 5vw, 40px) !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #58a6ff !important;
        font-size: clamp(18px, 4vw, 28px) !important;
    }
    
    h4 {
        color: #79c0ff !important;
        font-size: clamp(14px, 3vw, 18px) !important;
    }
    
    /* Input boxes */
    input, select, textarea {
        background-color: #161b22 !important;
        color: #e0e0e0 !important;
        border-color: #30363d !important;
        font-size: clamp(12px, 2vw, 14px) !important;
    }
    
    /* Responsive button styling */
    button {
        font-size: clamp(12px, 2vw, 14px) !important;
        padding: clamp(8px, 1vw, 12px) 16px !important;
    }
    
    /* Mobile adjustments */
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
        }
        
        .stSelectbox, .stTextInput {
            width: 100% !important;
        }
        
        [data-testid="metric-container"] {
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        
        button {
            width: 100% !important;
            margin-top: 8px !important;
        }
    }
    
    @media (max-width: 480px) {
        h1 {
            font-size: 20px !important;
        }
        
        h2, h3 {
            font-size: 16px !important;
        }
        
        /* Sidebar adjustments for mobile */
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
    
    /* General responsive spacing */
    .css-1d391kg {
        padding: 1rem 0.5rem !important;
    }
    
    /* Columns gap adjustment */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Alert styling */
    .stAlert {
        font-size: clamp(12px, 2vw, 14px) !important;
        padding: clamp(10px, 2vw, 15px) !important;
    }
    
    /* Text sizing */
    p, span, div {
        font-size: clamp(12px, 2vw, 14px) !important;
    }
    
    /* Divider styling */
    hr {
        margin: 1rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)
def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data (pd.Series): Series of closing prices
        period (int): Period for RSI calculation (default: 14)
    
    Returns:
        pd.Series: RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_moving_average(data, period):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data (pd.Series): Series of closing prices
        period (int): Period for SMA calculation
    
    Returns:
        pd.Series: Moving average values
    """
    return data.rolling(window=period).mean()


def fetch_stock_data(ticker, period="1y"):
    """
    Fetch stock data from yfinance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Period for data fetching (default: '1y' for 1 year)
    
    Returns:
        pd.DataFrame: Stock data or None if ticker is invalid
    """
    try:
        data = yf.download(ticker.upper(), period=period, progress=False)
        if data.empty or len(data) < 250:
            return None
        
        return data
    except Exception:
        return None


def prepare_training_data(df):
    """
    Prepare data for model training by calculating indicators and creating target.
    
    Args:
        df (pd.DataFrame): Stock data with OHLC values
    
    Returns:
        tuple: (X features, y target, indicators dataframe) or (None, None, None) if insufficient data
    """
    data = df.copy()
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    data['MA50'] = calculate_moving_average(data['Close'], period=50)
    data['MA200'] = calculate_moving_average(data['Close'], period=200)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()
    if len(data) < 50:
        return None, None, None
    X = data[['RSI', 'MA50', 'MA200']].values
    y = data['Target'].values
    
    return X, y, data


def train_model(X, y):
    """
    Train Logistic Regression model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
    
    Returns:
        tuple: (trained model, accuracy score)
    """
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy


def predict_next_day_trend(model, rsi, ma50, ma200):
    """
    Predict next day's trend.
    
    Args:
        model: Trained Logistic Regression model
        rsi (float): Current RSI value
        ma50 (float): Current 50-day MA
        ma200 (float): Current 200-day MA
    
    Returns:
        tuple: (trend prediction as string, confidence as float)
    """
    features = np.array([[rsi, ma50, ma200]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]
    
    trend = "📈 UPTREND" if prediction == 1 else "📉 DOWNTREND"
    confidence_value = max(confidence) * 100
    
    return trend, confidence_value


def create_price_chart(df):
    """
    Create interactive price chart with moving averages using Plotly.
    
    Args:
        df (pd.DataFrame): Stock data with indicators
    
    Returns:
        plotly graph object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#58a6ff', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> $%{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        mode='lines',
        name='50-day MA',
        line=dict(color='#ffa657', width=2, dash='dash'),
        hovertemplate='<b>50-day MA:</b> $%{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA200'],
        mode='lines',
        name='200-day MA',
        line=dict(color='#ff7b72', width=2, dash='dot'),
        hovertemplate='<b>200-day MA:</b> $%{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title='Stock Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_dark',
        height=max(300, min(600, 500)),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(14, 17, 23, 1)',
        plot_bgcolor='rgba(14, 17, 23, 1)',
        font=dict(color='#e0e0e0'),
    )
    
    return fig


def create_rsi_chart(df):
    """
    Create RSI indicator chart using Plotly.
    
    Args:
        df (pd.DataFrame): Stock data with RSI indicator
    
    Returns:
        plotly graph object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI (14)',
        line=dict(color='#79c0ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(121, 192, 255, 0.2)',
        hovertemplate='<b>RSI:</b> %{y:.2f}<extra></extra>'
    ))
    fig.add_hline(y=70, line_dash='dash', line_color='#ff7b72', 
                  annotation_text='Overbought (70)', annotation_position='right')
    fig.add_hline(y=30, line_dash='dash', line_color='#79c1ff', 
                  annotation_text='Oversold (30)', annotation_position='right')
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        hovermode='x unified',
        template='plotly_dark',
        height=max(250, min(450, 350)),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(14, 17, 23, 1)',
        plot_bgcolor='rgba(14, 17, 23, 1)',
        font=dict(color='#e0e0e0'),
        yaxis=dict(range=[0, 100])
    )
    
    return fig
def main():
    """Main Streamlit application."""
    st.title("📈 Stock Price Trend Predictor")
    st.markdown("""
        **Predict next day's stock trend using technical analysis and machine learning**
        
        This app uses RSI, 50-day MA, and 200-day MA to predict whether a stock will 
        go up or down tomorrow using Logistic Regression.
    """)
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Popular tickers dropdown
        popular_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "PYPL", "INTC"]
        selected_ticker = st.selectbox(
            "Select a Popular Ticker",
            options=popular_tickers,
            index=0,
            help="Choose from well-known stocks"
        )
        
        # Custom ticker input
        custom_ticker = st.text_input(
            "Or Enter a Custom Ticker",
            value="",
            placeholder="e.g., IBM, AMD, BABA",
            help="Type any valid stock ticker symbol"
        ).upper()
        
        # Use custom ticker if provided, otherwise use selected ticker
        ticker = custom_ticker if custom_ticker else selected_ticker
        
        fetch_button = st.button("📊 Fetch & Analyze", use_container_width=True)
        
        st.divider()
        st.markdown("**About**")
        st.markdown("""
            - **Data**: Last 1 year (252 trading days) of daily data
            - **Indicators**: RSI (14), MA50, MA200
            - **Model**: Logistic Regression
            - **Accuracy**: Tested on 20% holdout set
        """)
    if fetch_button or ticker:
        with st.spinner(f"📊 Fetching data for {ticker}..."):
            raw_data = fetch_stock_data(ticker)
            if raw_data is None:
                st.error(
                    f"❌ Invalid ticker or insufficient data: **{ticker}**\n\n"
                    "Please enter a valid stock ticker (e.g., AAPL, TSLA, GOOGL)"
                )
                return
        
        
        with st.spinner("🔧 Preparing data and training model..."):
            X, y, data = prepare_training_data(raw_data)
            
            if X is None:
                st.error("❌ Not enough data to train the model. Please try again.")
                return
            
            
            model, model_accuracy = train_model(X, y)
        
        # Get latest indicators
        latest_rsi = data['RSI'].iloc[-1].item()
        latest_ma50 = data['MA50'].iloc[-1].item()
        latest_ma200 = data['MA200'].iloc[-1].item()
        latest_close = data['Close'].iloc[-1].item()
        
        # Make prediction
        prediction, confidence = predict_next_day_trend(model, latest_rsi, latest_ma50, latest_ma200)
        
        # ================================================================
        # DISPLAY METRICS IN COLUMNS - RESPONSIVE LAYOUT
        # ================================================================
        st.subheader("📊 Key Metrics")
        
        # Create responsive columns (mobile: 1, tablet: 2, desktop: 5)
        metrics_data = [
            ("Current Price", f"${latest_close:.2f}", None),
            ("RSI (14)", f"{latest_rsi:.2f}", "Overbought" if latest_rsi > 70 else ("Oversold" if latest_rsi < 30 else "Neutral")),
            ("50-day MA", f"${latest_ma50:.2f}", f"${latest_close - latest_ma50:+.2f}"),
            ("200-day MA", f"${latest_ma200:.2f}", f"${latest_close - latest_ma200:+.2f}"),
            ("Model Accuracy", f"{model_accuracy * 100:.1f}%", "Based on test set")
        ]
        
        # Display metrics in responsive grid
        for i in range(0, len(metrics_data), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                if i + j < len(metrics_data):
                    label, value, delta = metrics_data[i + j]
                    with col:
                        st.metric(label=label, value=value, delta=delta)
        
        st.divider()
        
        # ================================================================
        # PREDICTION RESULT (HIGHLIGHTED)
        # ================================================================
        prediction_color = "#238636" if "UPTREND" in prediction else "#da3633"
        
        st.markdown(f"""
            <div style="
                background-color: {prediction_color}; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center;
                margin: 20px 0;
            ">
                <h2 style="color: white; margin: 0;">Next Day Prediction</h2>
                <h1 style="color: white; margin: 10px 0;">{prediction}</h1>
                <p style="color: white; margin: 0; font-size: 18px;">
                    Confidence: <strong>{confidence:.1f}%</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ================================================================
        # CHARTS
        # ================================================================
        st.subheader("📈 Technical Analysis")
        
        # Responsive columns for charts (1 column on mobile, 3:2 on desktop)
        chart_col1, chart_col2 = st.columns([3, 2], gap="medium")
        
        with chart_col1:
            # Price chart with MAs
            price_fig = create_price_chart(data.tail(100))  # Last 100 days
            st.plotly_chart(price_fig, use_container_width=True)
        
        with chart_col2:
            # RSI chart
            rsi_fig = create_rsi_chart(data.tail(100))  # Last 100 days
            st.plotly_chart(rsi_fig, use_container_width=True)
        
        st.divider()
        
        # ================================================================
        # DATA TABLE
        # ================================================================
        with st.expander("📋 View Recent Data", expanded=False):
            display_data = data[['Close', 'RSI', 'MA50', 'MA200', 'Target']].tail(10).copy()
            display_data = display_data.round(2)
            display_data.index.name = 'Date'
            st.dataframe(display_data, use_container_width=True)
        
        # ================================================================
        # INFORMATION CARDS
        # ================================================================
        st.subheader("ℹ️ Interpretation Guide")
        
        info_col1, info_col2, info_col3 = st.columns(3, gap="small")
        
        with info_col1:
            st.info("""
                **RSI (14)**
                - < 30: Oversold (potential buy)
                - 30-70: Neutral
                - > 70: Overbought (potential sell)
            """)
        
        with info_col2:
            st.info("""
                **Moving Averages**
                - Price > MA50 > MA200: Bullish
                - Price < MA50 < MA200: Bearish
                - Golden Cross: Buy signal
                - Death Cross: Sell signal
            """)
        
        with info_col3:
            st.warning("""
                **⚠️ Disclaimer**
                
                This is for educational purposes only. Not financial advice. Always do your own research before trading.
            """)


if __name__ == "__main__":
    main()
