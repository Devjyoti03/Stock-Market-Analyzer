# Stock Market Analyzer 📈

A comprehensive stock market analysis and visualization tool built with Streamlit that provides real-time data analysis, interactive charts, and powerful insights for stock market enthusiasts and investors.

## Features 🚀

- **Real-time Stock Data**: Fetch live stock prices and historical data
- **Interactive Charts**: Dynamic plotting with multiple chart types (candlestick, line, volume)
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
- **Portfolio Tracking**: Monitor multiple stocks simultaneously
- **Data Export**: Download analysis results in CSV format
- **Responsive Design**: Clean, intuitive web interface
- **Performance Metrics**: Calculate returns, volatility, and risk metrics

## Installation 💻

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Devjyoti03/Stock-Market-Analyzer.git
   cd Stock-Market-Analyzer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## Dependencies 📦

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
yfinance>=0.2.18
ta>=0.10.2
requests>=2.31.0
datetime
```

## Usage 📊

### Basic Analysis

1. **Select a Stock**: Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)
2. **Choose Time Period**: Select from 1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y
3. **Select Chart Type**: Choose between Line, Candlestick, or Volume charts
4. **Apply Technical Indicators**: Toggle on/off various technical analysis tools

### Advanced Features

- **Portfolio Analysis**: Add multiple stocks to track performance
- **Comparison Tool**: Compare up to 5 stocks side by side
- **Risk Assessment**: View volatility and risk metrics
- **Export Data**: Download historical data and analysis results

### Technical Indicators Available

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Volume Weighted Average Price (VWAP)
