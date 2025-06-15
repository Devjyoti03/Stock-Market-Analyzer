import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📈 Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def fetch_stock_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Exponential Moving Average
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    return data

def create_candlestick_chart(data, symbol):
    """Create an interactive candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{symbol} Stock Price', 'Volume', 'RSI'),
        vertical_spacing=0.05,
        row_width=[0.2, 0.1, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def perform_sentiment_analysis(symbol):
    """Simulate sentiment analysis (in real project, you'd use news APIs)"""
    # This is a placeholder - in a real project, you'd integrate with news APIs
    sentiment_score = np.random.uniform(-1, 1)
    if sentiment_score > 0.1:
        sentiment = "Positive"
        color = "green"
    elif sentiment_score < -0.1:
        sentiment = "Negative" 
        color = "red"
    else:
        sentiment = "Neutral"
        color = "gray"
    
    return sentiment, sentiment_score, color

def predict_price_trend(data, days=30):
    """Simple price prediction using polynomial regression"""
    # Prepare data
    data_clean = data.dropna()
    X = np.arange(len(data_clean)).reshape(-1, 1)
    y = data_clean['Close'].values
    
    # Use polynomial features
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future prices
    future_X = np.arange(len(data_clean), len(data_clean) + days).reshape(-1, 1)
    future_X_poly = poly_features.transform(future_X)
    predictions = model.predict(future_X_poly)
    
    return predictions

# Main application
def main():
    st.markdown('<h1 class="main-header">📈 Stock Market Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Stock selection
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=default_stocks + ['Custom'],
        index=0
    )
    
    if selected_symbol == 'Custom':
        selected_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    # Analysis type
    analysis_type = st.sidebar.multiselect(
        "Select Analysis Types",
        options=['Technical Analysis', 'Price Prediction', 'Sentiment Analysis', 'Financial Metrics'],
        default=['Technical Analysis', 'Financial Metrics']
    )
    
    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        with st.spinner(f"Fetching data for {selected_symbol}..."):
            # Fetch data
            data, info = fetch_stock_data(selected_symbol, time_period)
            
            if data is not None and not data.empty:
                # Calculate technical indicators
                data = calculate_technical_indicators(data)
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.2f}",
                        delta=f"{change:.2f} ({change_pct:.2f}%)"
                    )
                
                with col2:
                    st.metric(
                        label="52W High",
                        value=f"${data['High'].max():.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="52W Low", 
                        value=f"${data['Low'].min():.2f}"
                    )
                
                with col4:
                    avg_volume = data['Volume'].mean()
                    st.metric(
                        label="Avg Volume",
                        value=f"{avg_volume:,.0f}"
                    )
                
                # Technical Analysis
                if 'Technical Analysis' in analysis_type:
                    st.subheader("📊 Technical Analysis")
                    fig = create_candlestick_chart(data, selected_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # MACD Chart
                    st.subheader("📈 MACD Analysis")
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')))
                    fig_macd.update_layout(title="MACD Indicator", height=400)
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Price Prediction
                if 'Price Prediction' in analysis_type:
                    st.subheader("🔮 Price Prediction (Next 30 Days)")
                    predictions = predict_price_trend(data, 30)
                    
                    # Create prediction chart
                    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='D')
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=data.index[-60:], 
                        y=data['Close'].iloc[-60:], 
                        name='Historical', 
                        line=dict(color='blue')
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates, 
                        y=predictions, 
                        name='Predicted', 
                        line=dict(color='red', dash='dash')
                    ))
                    fig_pred.update_layout(title="Price Prediction", height=400)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.info(f"Predicted price in 30 days: ${predictions[-1]:.2f}")
                
                # Sentiment Analysis
                if 'Sentiment Analysis' in analysis_type:
                    st.subheader("💭 Market Sentiment")
                    sentiment, sentiment_score, color = perform_sentiment_analysis(selected_symbol)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Overall Sentiment:** :{color}[{sentiment}]")
                        st.progress(abs(sentiment_score))
                    
                    with col2:
                        st.markdown(f"**Sentiment Score:** {sentiment_score:.3f}")
                        st.caption("Score ranges from -1 (very negative) to +1 (very positive)")
                
                # Financial Metrics
                if 'Financial Metrics' in analysis_type:
                    st.subheader("💰 Financial Metrics")
                    
                    # Calculate metrics
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Volatility (Annual)", f"{volatility:.2%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                    
                    with col4:
                        current_rsi = data['RSI'].iloc[-1]
                        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                        st.metric("Current RSI", f"{current_rsi:.1f} ({rsi_signal})")
                    
                    # Returns distribution
                    st.subheader("📊 Returns Distribution")
                    fig_hist = px.histogram(
                        returns, 
                        nbins=50, 
                        title="Daily Returns Distribution",
                        labels={'value': 'Daily Returns', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Raw data
                with st.expander("📋 View Raw Data"):
                    st.dataframe(data.tail(50))
                    
                    # Download button
                    csv = data.to_csv()
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"{selected_symbol}_stock_data.csv",
                        mime="text/csv"
                    )
            
            else:
                st.error(f"Could not fetch data for symbol: {selected_symbol}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            📈 Stock Market Analyzer | Built by Dev with Streamlit & Python<br>
            <small>Disclaimer: This is for educational purposes only.</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
