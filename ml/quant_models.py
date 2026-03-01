"""
Advanced Quantitative Finance Engine (V3)

Core mathematics for:
1. Kelly Criterion (Optimal Sizing)
2. Regime Detection (Gaussian Mixture proxy for HMM)
3. Risk Parity (Inverse Volatility Portfolio Allocation)
4. Conditional Value at Risk (CVaR)
5. Statistical Arbitrage (Pairs Spread Z-Score)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, pearsonr
import statsmodels.tsa.stattools as ts

# -------------------------------------------------------------------
# 1. Kelly Criterion
# -------------------------------------------------------------------
def calculate_kelly_fraction(win_rate: float, win_loss_ratio: float, fraction_multiplier: float = 0.5) -> float:
    """
    Calculates the optimal fraction of bankroll to wager.
    Formula: K = W - ((1 - W) / R)
    Where W is win rate, R is Average Win / Average Loss ratio.
    Default uses "Half-Kelly" (multiplier=0.5) for safer variance.
    """
    if win_loss_ratio <= 0.0 or win_rate <= 0.0:
        return 0.0
        
    k = win_rate - ((1.0 - win_rate) / win_loss_ratio)
    
    # Bound between 0 (don't trade) and 1 (all in)
    k = max(0.0, min(1.0, k))
    
    return k * fraction_multiplier

# -------------------------------------------------------------------
# 2. Regime Detection (HMM Proxy via GMM)
# -------------------------------------------------------------------
def estimate_regime(returns: np.ndarray) -> str:
    """
    Uses a 2-component Gaussian Mixture Model to classify the current market regime
    based on a rolling window of returns.
    Returns: 'BULL', 'BEAR', or 'CHOPPY'
    """
    if len(returns) < 50:
        return 'UNKNOWN'
        
    # We need a 2D array for sklearn
    X = returns.reshape(-1, 1)
    
    try:
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        
        # Determine which component is high-vol and low-vol
        vols = np.sqrt(gmm.covariances_.flatten())
        means = gmm.means_.flatten()
        
        # Predict the regime of the *most recent* return
        current_state = gmm.predict(X[-1:])[0]
        
        current_vol = vols[current_state]
        current_mean = means[current_state]
        
        # Simple heuristic mapping
        if current_mean > 0 and current_vol < np.median(vols)*1.5:
            return 'BULL'
        elif current_mean < 0 and current_vol > np.median(vols):
            return 'BEAR'
        else:
            return 'CHOPPY'
            
    except Exception:
        return 'UNKNOWN'

# -------------------------------------------------------------------
# 3. Risk Parity (Inverse Volatility Allocation)
# -------------------------------------------------------------------
def calculate_risk_parity_weights(returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates portfolio weights such that each asset contributes equally to risk.
    Uses inverse-variance weighting as a robust proxy for full risk parity.
    returns_df: DataFrame where columns are symbols, rows are returns.
    """
    std_devs = returns_df.std()
    
    # Avoid div-by-zero
    std_devs = std_devs.replace(0, np.nan).dropna()
    
    if std_devs.empty:
        return {col: 1.0/len(returns_df.columns) for col in returns_df.columns}
        
    inv_vars = 1.0 / (std_devs ** 2)
    weights = inv_vars / inv_vars.sum()
    
    return weights.to_dict()

# -------------------------------------------------------------------
# 4. Conditional Value at Risk (CVaR) / Expected Shortfall
# -------------------------------------------------------------------
def calculate_cvar(portfolio_returns: np.ndarray, confidence_level: float = 0.99) -> float:
    """
    Calculates the Expected Shortfall (CVaR) at a given confidence level.
    What is the average expected loss *if* the loss exceeds the VaR threshold?
    Returns a positive float representing percentage loss (e.g., 0.05 = 5% loss).
    """
    if len(portfolio_returns) < 20:
        return 0.0
        
    # Find the Value at Risk (VaR) threshold (e.g., the worst 1% of returns)
    var_threshold = np.percentile(portfolio_returns, (1.0 - confidence_level) * 100)
    
    # Get all returns worse than the VaR threshold
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return 0.0
        
    # CVaR is the expectation (mean) of the tail loss
    cvar = -np.mean(tail_returns)
    return max(0.0, float(cvar))

# -------------------------------------------------------------------
# 5. Statistical Arbitrage (Pairs Spread Z-Score)
# -------------------------------------------------------------------
def calculate_spread_zscore(price_series_1: np.ndarray, price_series_2: np.ndarray, window: int = 20) -> float:
    """
    Calculates the current Z-score of the spread ratio between two assets.
    If Z > 2, Asset 1 is historically overvalued relative to Asset 2.
    If Z < -2, Asset 1 is historically undervalued relative to Asset 2.
    """
    if len(price_series_1) != len(price_series_2) or len(price_series_1) < window:
        return 0.0
        
    # Calculate ratio spread
    spreads = price_series_1 / price_series_2
    
    # Get the rolling window
    recent_spreads = spreads[-window:]
    
    mean = np.mean(recent_spreads)
    std = np.std(recent_spreads)
    
    if std == 0:
        return 0.0
        
    current_spread = spreads[-1]
    z_score = (current_spread - mean) / std
    
    return float(z_score)

def check_cointegration(series1: np.ndarray, series2: np.ndarray, p_value_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Checks if two price series are cointegrated using the Engle-Granger two-step method.
    If cointegrated, they tend to move together over time, making them suitable for Pairs Trading.
    Returns (is_cointegrated, p_value).
    """
    if len(series1) < 30 or len(series2) < 30 or len(series1) != len(series2):
        return False, 1.0
        
    try:
        # We need statsmodels for the ADF test on residuals
        # statsmodels.tsa.stattools.coint is the standard
        score, p_value, _ = ts.coint(series1, series2)
        return p_value < p_value_threshold, p_value
    except Exception:
        # Fallback to simple correlation if statsmodels fails or isn't available
        # This is NOT true cointegration, but a poor man's proxy for environments without statsmodels
        corr, p_value = pearsonr(series1, series2)
        is_correlated = corr > 0.8 and p_value < p_value_threshold
        return is_correlated, p_value

