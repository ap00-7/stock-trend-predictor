# 📈 Stock Price Trend Predictor

A modern Streamlit web app that predicts whether a stock's price will go up or down tomorrow using technical indicators and machine learning.

## 🎯 Features

- **Real-time Stock Data**: Fetches 6 months of historical data using yfinance
- **Technical Indicators**:
  - RSI (Relative Strength Index) - 14 day period
  - 50-day Moving Average
  - 200-day Moving Average
- **Machine Learning**: Trains a Logistic Regression model to predict next day trend
- **Interactive Charts**: Plotly visualizations with dark theme
- **Model Accuracy**: Displays accuracy metrics on test data
- **Error Handling**: Validates ticker symbols and handles edge cases
- **Modern UI**: Clean, responsive design with Streamlit columns

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Navigate to project directory**:
   ```bash
   cd path/to/project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run stock_predictor.py
   ```

5. **Open in browser**:
   - The app will automatically open in your default browser at `http://localhost:8501`
   - If not, manually go to that URL

## 📊 How to Use

1. **Enter Stock Ticker**: Type a valid stock ticker (e.g., AAPL, TSLA, GOOGL, MSFT)
2. **Click "Fetch & Analyze"**: The app will:
   - Download 6 months of stock data
   - Calculate technical indicators
   - Train the ML model
   - Make a prediction
3. **View Results**:
   - See key metrics (Current Price, RSI, Moving Averages)
   - Check the prediction: **UPTREND** or **DOWNTREND**
   - View confidence level
   - Analyze interactive charts

## 📈 Technical Details

### Data Processing
- **Source**: yfinance library (Yahoo Finance API)
- **Period**: Last 6 months of daily data
- **Minimum Data**: At least 200 data points required for MA200 calculation

### Indicators Calculated

**RSI (Relative Strength Index)**
- Measures momentum and overbought/oversold conditions
- Range: 0-100
- <30: Oversold | >70: Overbought | 30-70: Neutral

**Moving Averages**
- **MA50**: Average of last 50 closing prices (short-term trend)
- **MA200**: Average of last 200 closing prices (long-term trend)

### Prediction Logic

**Target Variable**:
- 1 if next day's close > today's close (UPTREND)
- 0 if next day's close ≤ today's close (DOWNTREND)

**Model**:
- Algorithm: Logistic Regression
- Features: [RSI, MA50, MA200]
- Train/Test Split: 80/20
- Accuracy: Displayed on the app

## 🔍 Example Usage

```
Input: AAPL
↓
Fetches Apple's stock data
↓
Calculates RSI = 65.4, MA50 = $180.23, MA200 = $175.50
↓
Trains model on historical data
↓
Predicts: 📈 UPTREND (Confidence: 72.3%)
↓
Shows accuracy: 58.2%
```

## ⚠️ Important Notes

- **Educational Purpose**: This app is for learning and demonstration only
- **Not Financial Advice**: Do not make investment decisions based solely on this predictor
- **Market Risk**: Past performance doesn't guarantee future results
- **Limited Model**: Technical indicators alone are not sufficient for investment decisions
- **Context Matters**: Always consider news, economic factors, and broader market conditions

## 📦 Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API client
- **scikit-learn**: Machine learning library
- **plotly**: Interactive visualizations

## 🐛 Troubleshooting

### "Invalid ticker or insufficient data"
- Ensure the ticker symbol is correct (e.g., AAPL, not APP)
- The stock must have at least 200 days of historical data
- Some penny stocks might not have enough data

### "ModuleNotFoundError"
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Verify your virtual environment is activated
- Try reinstalling: `pip install --upgrade -r requirements.txt`

### "No data fetched"
- Check your internet connection
- Yahoo Finance might be temporarily unavailable
- Try again in a few moments

### App runs slowly
- This is normal on first run (model training takes time)
- Subsequent predictions with same ticker will be faster
- Allow 2-3 seconds for data fetching and model training

## 📝 Customization Ideas

You can enhance this app by:
- Adding more technical indicators (Bollinger Bands, MACD, ATR)
- Using different ML models (Random Forest, XGBoost, Neural Networks)
- Adding sentiment analysis from news/social media
- Including volume analysis
- Adding multiple timeframe analysis
- Creating portfolio tracker for multiple stocks
- Adding backtesting functionality
- Implementing real-time updates

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your Python version is 3.8+
4. Check internet connection for data fetching

## 📄 License

This project is provided as-is for educational purposes.

---

**Happy Predicting! 📊📈**
