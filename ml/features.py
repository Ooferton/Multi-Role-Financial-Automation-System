import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Library of technical indicators calculated using Pandas.
    Designed to be robust and efficient for RL feature engineering.
    """
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a standard suite of technical indicators to the dataframe.
        Expects 'close', 'high', 'low', 'volume' columns.
        """
        df = df.copy()
        
        # Trend
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['sma_200'] = TechnicalIndicators.sma(df['close'], 200)
        
        # Momentum
        df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        macd, signal, histogram = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram  # NEW for v2
        
        # Volatility
        df['atr_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
        df['bb_upper'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(df['close'], 20, 2)
        
        # Volume
        if 'volume' in df.columns:
            df['obv'] = TechnicalIndicators.on_balance_volume(df['close'], df['volume'])
        else:
            df['obv'] = 0.0
            
        # Comparison / Relative features (Better for ML)
        df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Normalized features for v2 (ML-friendly)
        df['atr_norm'] = df['atr_14'] / df['close']  # ATR as % of price
        obv_mean = df['obv'].rolling(window=20).mean()
        obv_std = df['obv'].rolling(window=20).std()
        df['obv_norm'] = (df['obv'] - obv_mean) / (obv_std + 1e-8)  # Z-scored OBV
        
        # Fill NaNs (important for RL start)
        df.fillna(0, inplace=True)
        
        return df

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return pd.Series(true_range).rolling(window=window).mean()

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
