import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from ml.regime_detector import RegimeDetector
import itertools

DB_PATH = "data/feature_store.db"

def load_historical_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """Loads historical data for a given symbol and resamples to OHLC if needed."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    # The system primarily saves to 'ticks'. We resample them to 5m intervals for analysis.
    query = f"SELECT timestamp, price FROM ticks WHERE symbol='{symbol}' AND timestamp > datetime('now', '-{days} days') ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty or len(df) < 5:
        return pd.DataFrame()
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Resample to 5-minute OHLC to get High/Low/Close for analysis
    resampled = df['price'].resample('5min').ohlc().dropna()
    resampled = resampled.reset_index()
    
    return resampled

def run_regime_detection(symbol: str = "SPY", days: int = 180) -> pd.DataFrame:
    """Runs the HMM regime detector over historical data for visualization."""
    df = load_historical_data(symbol, days)
    if df.empty or len(df) < 50:
        if not df.empty:
            df['regime'] = "UNKNOWN"
        return df
    
    # Calculate features needed for Regime Detector
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=14).std()
    df = df.dropna().copy()
    
    features = df[['returns', 'volatility']].values
    detector = RegimeDetector(n_regimes=3)
    detector.train(features)
    
    # Predict regimes
    regimes = []
    for f in features:
        regimes.append(detector.predict_regime(f))
    df['regime'] = regimes
    return df

def run_factor_ranking(symbols: list, days: int = 90) -> pd.DataFrame:
    """Ranks a list of symbols based on Momentum, Volatility, and Trend."""
    results = []
    
    for sym in symbols:
        df = load_historical_data(sym, days)
        if df.empty or len(df) < 30:
            continue
            
        current_price = df['close'].iloc[-1]
        past_price = df['close'].iloc[0]
        
        # Momentum: % Return over the period
        momentum = (current_price - past_price) / past_price
        
        # Volatility: Average daily true range (simplified)
        volatility = (df['high'] - df['low']).mean() / current_price
        
        # Trend: Distance from 20-period SMA
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        trend = (current_price - sma_20) / sma_20 if not np.isnan(sma_20) else 0.0
        
        results.append({
            "Symbol": sym,
            "Price": round(current_price, 2),
            "Momentum (%)": round(momentum * 100, 2),
            "Volatility (%)": round(volatility * 100, 2),
            "Trend (vs SMA20)": round(trend * 100, 2),
            "Score": round((momentum * 0.6) + (trend * 0.4) - (volatility * 0.5), 4) # Arbitrary factor weights
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Normalize score to 0-100 gauge
        min_score = res_df['Score'].min()
        max_score = res_df['Score'].max()
        if max_score > min_score:
            res_df['Conviction (0-100)'] = ((res_df['Score'] - min_score) / (max_score - min_score) * 100).round(1)
        else:
            res_df['Conviction (0-100)'] = 50.0
        res_df = res_df.sort_values(by="Conviction (0-100)", ascending=False).reset_index(drop=True)
        res_df = res_df.drop(columns=["Score"])
    return res_df

def run_cointegration_scan(symbols: list, days: int = 180) -> pd.DataFrame:
    """Scans combinations of symbols for statistical cointegration (Pairs Trading)."""
    # Requires statsmodels. We'll use a simplified correlation + distance metric 
    # if statsmodels isn't installed. Let's try to import it.
    try:
        from statsmodels.tsa.stattools import coint
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False

    data_dict = {}
    for sym in symbols:
        df = load_historical_data(sym, days)
        if len(df) > 50:
            # Resample to daily to ensure alignment
            df.set_index('timestamp', inplace=True)
            daily = df['close'].resample('D').last().dropna()
            data_dict[sym] = daily
            
    if len(data_dict) < 2:
        return pd.DataFrame()
        
    prices_df = pd.DataFrame(data_dict).dropna()
    valid_symbols = prices_df.columns.tolist()
    
    results = []
    pairs = list(itertools.combinations(valid_symbols, 2))
    
    for s1, s2 in pairs:
        p1 = prices_df[s1]
        p2 = prices_df[s2]
        
        correlation = p1.corr(p2)
        
        # Calculate z-score of the spread
        spread = p1 - p2
        z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
        
        coint_pvalue = 1.0
        if HAS_STATSMODELS and correlation > 0.5:
            try:
                score, pval, _ = coint(p1, p2)
                coint_pvalue = pval
            except:
                pass
                
        # We look for high correlation, low p-value (cointegrated), and actionable z-score
        results.append({
            "Pair": f"{s1} / {s2}",
            "Correlation": round(correlation, 3),
            "Cointegration P-Val": round(coint_pvalue, 4) if HAS_STATSMODELS else "N/A",
            "Spread Z-Score": round(z_score, 2),
            "Signal": "SHORT SPREAD" if z_score > 2 else ("LONG SPREAD" if z_score < -2 else "NEUTRAL")
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by most actionable z-scores
        res_df['Abs Z'] = res_df['Spread Z-Score'].abs()
        res_df = res_df.sort_values(by="Abs Z", ascending=False).drop(columns=["Abs Z"]).reset_index(drop=True)
    return res_df

def run_backtest(strategy_name: str, symbol: str, days: int = 180) -> dict:
    """Simulates a trading strategy over historical data and returns performance metrics."""
    df = load_historical_data(symbol, days)
    if df.empty or len(df) < 50:
        return {"error": "Not enough historical data"}
        
    df['returns'] = df['close'].pct_change()
    
    if strategy_name == "MACD Crossover":
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Position: 1 if MACD > Signal (Bullish), else 0
        df['position'] = np.where(macd > signal, 1, 0)
        df['position'] = df['position'].shift(1).fillna(0) # Shift to avoid lookahead
        
    elif strategy_name == "Mean Reversion (Bollinger)":
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        df['signal'] = np.nan
        df.loc[df['close'] < lower, 'signal'] = 1   # Buy oversold
        df.loc[df['close'] > upper, 'signal'] = -1  # Short overbought
        df['position'] = df['signal'].ffill().shift(1).fillna(0) # Hold until cross
        
    else:  # Buy and Hold Baseline
        df['position'] = 1
        
    df['strategy_returns'] = df['position'] * df['returns']
    df['cumulative_market'] = (1 + df['returns']).cumprod() * 10000
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod() * 10000
    
    total_return = (df['cumulative_strategy'].iloc[-1] - 10000) / 10000 * 100
    market_return = (df['cumulative_market'].iloc[-1] - 10000) / 10000 * 100
    
    winning_days = (df['strategy_returns'] > 0).sum()
    losing_days = (df['strategy_returns'] < 0).sum()
    win_rate = (winning_days / (winning_days + losing_days)) * 100 if (winning_days + losing_days) > 0 else 0
    trades_taken = int((df['position'].diff() != 0).sum())
    
    out_df = df[['timestamp', 'close', 'cumulative_market', 'cumulative_strategy']].dropna()
    
    return {
        "df": out_df,
        "total_return": round(total_return, 2),
        "market_return": round(market_return, 2),
        "win_rate": round(win_rate, 1),
        "trades_taken": trades_taken
    }

