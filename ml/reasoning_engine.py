import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class TradeReasoning:
    """Structured reasoning output for any financial decision."""
    summary: str                          # One-line verdict
    factors: List[str]                    # Bullet-point reasons
    confidence: float                     # 0-1 score
    risk_notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        factors_str = " • ".join(self.factors)
        risk_str = " | ".join(self.risk_notes) if self.risk_notes else "None"
        return (
            f"[{self.confidence:.0%} confidence] {self.summary}\n"
            f"  Factors: {factors_str}\n"
            f"  Risks: {risk_str}"
        )

@dataclass
class JournalEntry:
    """A higher-level reflection on the market state and model's 'feelings'."""
    sentiment: str                        # Market Vibe (Bullish/Bearish/Chaos)
    mood: str                             # Model Vibe (Aggressive/Cautious/Confused)
    observations: List[str]               # Broad market observations
    timestamp: datetime = field(default_factory=datetime.now)

    def to_markdown(self):
        obs_str = "\n".join([f"- {o}" for o in self.observations])
        return (
            f"### AI Journal Entry: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Market Sentiment:** {self.sentiment}\n"
            f"**Model Internal Mood:** {self.mood}\n\n"
            f"**Observations:**\n{obs_str}\n\n"
            f"---\n"
        )

class ReasoningEngine:
    """
    Generates human-readable explanations for AI decisions.
    Uses rule-based analysis of technical indicators, sentiment, and portfolio state.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def explain_trade(self, action_val: float, indicators: Dict, portfolio: Dict, symbol: str) -> TradeReasoning:
        """
        Generates reasoning for a trade decision.
        
        Args:
            action_val: RL agent output (-1 to 1)
            indicators: Dict with keys like rsi_14, macd, macd_signal, bb_width, dist_sma_20
            portfolio: Dict with keys like cash, position_qty
            symbol: Ticker symbol
        """
        factors = []
        risk_notes = []
        
        # Direction
        direction = "BUY" if action_val > 0 else "SELL" if action_val < 0 else "HOLD"
        conviction = abs(action_val)
        
        # --- Momentum Analysis ---
        rsi = indicators.get('rsi_14', 50)
        if rsi < 30:
            factors.append(f"RSI at {rsi:.0f} (oversold — bounce likely)")
        elif rsi > 70:
            factors.append(f"RSI at {rsi:.0f} (overbought — pullback risk)")
        elif 40 <= rsi <= 60:
            factors.append(f"RSI neutral at {rsi:.0f}")
        elif rsi < 40:
            factors.append(f"RSI weak at {rsi:.0f} (approaching oversold)")
        else:
            factors.append(f"RSI elevated at {rsi:.0f} (approaching overbought)")
        
        # --- MACD Analysis ---
        macd = indicators.get('macd', 0)
        signal = indicators.get('macd_signal', 0)
        if macd > signal and macd > 0:
            factors.append("MACD bullish crossover (strong upward momentum)")
        elif macd > signal:
            factors.append("MACD turning bullish (early momentum shift)")
        elif macd < signal and macd < 0:
            factors.append("MACD bearish crossover (strong downward momentum)")
        elif macd < signal:
            factors.append("MACD turning bearish (momentum fading)")
        
        # --- Trend Analysis ---
        sma_dist = indicators.get('dist_sma_20', 0)
        if sma_dist > 0.02:
            factors.append(f"Price {sma_dist:.1%} above 20-SMA (strong uptrend)")
        elif sma_dist > 0:
            factors.append(f"Price slightly above 20-SMA (mild uptrend)")
        elif sma_dist < -0.02:
            factors.append(f"Price {abs(sma_dist):.1%} below 20-SMA (downtrend)")
        else:
            factors.append(f"Price near 20-SMA (consolidation)")
        
        # --- Volatility Analysis ---
        bb_width = indicators.get('bb_width', 0)
        if bb_width > 0.05:
            risk_notes.append(f"High volatility (BB width: {bb_width:.2%})")
        elif bb_width < 0.02:
            factors.append("Low volatility — breakout may be imminent")
        
        # --- Position Risk ---
        cash = portfolio.get('cash', 0)
        position_qty = portfolio.get('position_qty', 0)
        if direction == "BUY" and cash < 10:
            risk_notes.append("Very low cash reserves")
        if direction == "SELL" and position_qty <= 0:
            risk_notes.append("No position to sell — short selling")
        if conviction < 0.3:
            risk_notes.append(f"Low conviction ({conviction:.0%}) — weak signal")
        
        # --- Build Summary ---
        if direction == "HOLD":
            summary = f"HOLD {symbol}: No strong signal detected"
        else:
            strength = "Strong" if conviction > 0.6 else "Moderate" if conviction > 0.3 else "Weak"
            # Pick the most important factor
            primary_factor = factors[0] if factors else "General market conditions"
            summary = f"{strength} {direction} {symbol}: {primary_factor}"
        
        return TradeReasoning(
            summary=summary,
            factors=factors,
            confidence=conviction,
            risk_notes=risk_notes
        )

    def explain_rebalance(self, current_allocation: Dict, target_allocation: Dict, reason: str) -> TradeReasoning:
        """Reasoning for wealth management rebalancing decisions."""
        factors = []
        risk_notes = []
        
        for asset, target_pct in target_allocation.items():
            current_pct = current_allocation.get(asset, 0)
            diff = target_pct - current_pct
            if abs(diff) > 0.01:
                direction = "Increasing" if diff > 0 else "Reducing"
                factors.append(f"{direction} {asset} by {abs(diff):.1%} ({current_pct:.1%} → {target_pct:.1%})")
        
        if not factors:
            factors.append("Portfolio already aligned with targets")
            
        return TradeReasoning(
            summary=f"Rebalance: {reason}",
            factors=factors,
            confidence=0.8,
            risk_notes=risk_notes
        )

    def explain_refinance(self, current_rate: float, new_rate: float, 
                          monthly_savings: float, break_even_months: int) -> TradeReasoning:
        """Reasoning for lending/refinance recommendations."""
        factors = [
            f"Current rate: {current_rate:.2%} → New rate: {new_rate:.2%}",
            f"Monthly savings: ${monthly_savings:,.2f}",
            f"Break-even in {break_even_months} months"
        ]
        
        risk_notes = []
        if break_even_months > 36:
            risk_notes.append("Long break-even period — consider if you'll stay that long")
        if new_rate > 0.06:
            risk_notes.append("Rate still elevated by historical standards")
        
        savings_annual = monthly_savings * 12
        confidence = min(0.95, (current_rate - new_rate) * 10)  # Higher confidence with bigger rate drop
        
        return TradeReasoning(
            summary=f"Refinance recommended: Save ${savings_annual:,.0f}/year",
            factors=factors,
            confidence=confidence,
            risk_notes=risk_notes
        )

    def explain_trade_v2(self, action_val: float, indicators: Dict, portfolio: Dict, symbol: str, 
                         sentiment: Optional[Dict] = None, macro: Optional[Dict] = None) -> TradeReasoning:
        """
        Generates reasoning for a v2 trade decision using Indicators + Sentiment + Macro.
        
        Args:
            action_val: RL agent output (-1 to 1)
            indicators: Dict with 10 indicator keys
            portfolio: Dict with keys like cash, position_qty
            symbol: Ticker symbol
            sentiment: Optional dict with 'score', 'verdict', 'headlines'
            macro: Optional dict with 'vibe', 'metrics', 'summary'
        """
        factors = []
        risk_notes = []
        
        # --- 0. Macro Analysis (The Economist Layer) ---
        if macro:
            vibe = macro.get('vibe', 'Neutral')
            factors.append(f"Economic Context: {vibe}")
            if "Crisis" in vibe or "Risk-Off" in vibe:
                risk_notes.append(f"Global macro alert: {vibe}")
        
        # --- 0. Sentiment Analysis (The Hearing Layer) ---
        if sentiment:
            score = sentiment.get('score', 0.0)
            verdict = sentiment.get('verdict', 'Neutral')
            if abs(score) > 0.1:
                factors.append(f"News Sentiment is {verdict} (Score: {score})")
                if score < -0.2:
                    risk_notes.append("Negative news flow detected — proceed with caution")
        
        # Direction
        direction = "BUY" if action_val > 0 else "SELL" if action_val < 0 else "HOLD"
        conviction = abs(action_val)
        
        # --- 1. RSI Analysis ---
        rsi = indicators.get('rsi_14', 50)
        if rsi < 30:
            factors.append(f"RSI at {rsi:.0f} (oversold — bounce likely)")
        elif rsi > 70:
            factors.append(f"RSI at {rsi:.0f} (overbought — pullback risk)")
        elif 40 <= rsi <= 60:
            factors.append(f"RSI neutral at {rsi:.0f}")
        elif rsi < 40:
            factors.append(f"RSI weak at {rsi:.0f} (approaching oversold)")
        else:
            factors.append(f"RSI elevated at {rsi:.0f} (approaching overbought)")
        
        # --- 2-3. MACD + Signal Analysis ---
        macd = indicators.get('macd', 0)
        signal = indicators.get('macd_signal', 0)
        if macd > signal and macd > 0:
            factors.append("MACD bullish crossover (strong upward momentum)")
        elif macd > signal:
            factors.append("MACD turning bullish (early momentum shift)")
        elif macd < signal and macd < 0:
            factors.append("MACD bearish crossover (strong downward momentum)")
        elif macd < signal:
            factors.append("MACD turning bearish (momentum fading)")
        
        # --- 4. BB Width (Volatility) ---
        bb_width = indicators.get('bb_width', 0)
        if bb_width > 0.05:
            risk_notes.append(f"High volatility (BB width: {bb_width:.2%})")
        elif bb_width < 0.02:
            factors.append("Low volatility — breakout may be imminent")
        
        # --- 5. Dist SMA 20 (Trend) ---
        sma_dist = indicators.get('dist_sma_20', 0)
        if sma_dist > 0.02:
            factors.append(f"Price {sma_dist:.1%} above 20-SMA (strong uptrend)")
        elif sma_dist > 0:
            factors.append(f"Price slightly above 20-SMA (mild uptrend)")
        elif sma_dist < -0.02:
            factors.append(f"Price {abs(sma_dist):.1%} below 20-SMA (downtrend)")
        else:
            factors.append(f"Price near 20-SMA (consolidation)")
        
        # --- 6. MACD Histogram (NEW v2) ---
        histogram = indicators.get('macd_histogram', 0)
        if histogram > 0 and macd > signal:
            factors.append(f"MACD histogram expanding ({histogram:.4f}) — momentum accelerating")
        elif histogram < 0 and macd < signal:
            factors.append(f"MACD histogram contracting ({histogram:.4f}) — momentum weakening")
        elif abs(histogram) < 0.01:
            factors.append("MACD histogram flat — no momentum divergence")
        
        # --- 7. BB Position (NEW v2) ---
        bb_pos = indicators.get('bb_position', 0.5)
        if bb_pos > 0.95:
            risk_notes.append(f"Price at upper Bollinger Band ({bb_pos:.0%}) — potential reversal")
        elif bb_pos < 0.05:
            factors.append(f"Price at lower Bollinger Band ({bb_pos:.0%}) — potential mean reversion")
        elif 0.4 <= bb_pos <= 0.6:
            factors.append(f"Price centered in Bollinger Bands ({bb_pos:.0%})")
        
        # --- 8. Dist SMA 50 (NEW v2) ---
        sma50_dist = indicators.get('dist_sma_50', 0)
        if sma50_dist > 0.03:
            factors.append(f"Price {sma50_dist:.1%} above 50-SMA (strong medium-term trend)")
        elif sma50_dist < -0.03:
            factors.append(f"Price {abs(sma50_dist):.1%} below 50-SMA (medium-term weakness)")
        
        # --- 9. ATR Normalized (NEW v2) ---
        atr_norm = indicators.get('atr_norm', 0)
        if atr_norm > 0.03:
            risk_notes.append(f"High ATR ({atr_norm:.1%} of price) — large daily swings")
        elif atr_norm < 0.005:
            factors.append(f"Low ATR ({atr_norm:.1%}) — quiet market, tight ranges")
        
        # --- 10. OBV Normalized (NEW v2) ---
        obv_norm = indicators.get('obv_norm', 0)
        if obv_norm > 1.5:
            factors.append(f"OBV surging ({obv_norm:.1f}σ above mean) — strong volume confirmation")
        elif obv_norm < -1.5:
            risk_notes.append(f"OBV dropping ({obv_norm:.1f}σ below mean) — volume divergence")
        
        # --- Position Risk ---
        cash = portfolio.get('cash', 0)
        position_qty = portfolio.get('position_qty', 0)
        if direction == "BUY" and cash < 10:
            risk_notes.append("Very low cash reserves")
        if direction == "SELL" and position_qty <= 0:
            risk_notes.append("No position to sell — short selling")
        if conviction < 0.3:
            risk_notes.append(f"Low conviction ({conviction:.0%}) — weak signal")
        
        # --- Build Summary ---
        if direction == "HOLD":
            summary = f"HOLD {symbol}: No strong signal detected"
        else:
            strength = "Strong" if conviction > 0.6 else "Moderate" if conviction > 0.3 else "Weak"
            primary_factor = factors[0] if factors else "General market conditions"
            summary = f"{strength} {direction} {symbol}: {primary_factor}"
        
        return TradeReasoning(
            summary=summary,
            factors=factors,
            confidence=conviction,
            risk_notes=risk_notes
        )

    def generate_journal_entry(self, indicators_agg: Dict, performance: Dict, 
                               news_agg: Optional[List[str]] = None, macro: Optional[Dict] = None) -> JournalEntry:
        """
        Reflects on the global state of the market across symbols.
        """
        volatility = indicators_agg.get('avg_volatility', 0)
        trend = indicators_agg.get('avg_trend', 0)
        win_rate = performance.get('win_rate', 0.5)
        
        # Determine Mood
        if win_rate > 0.6:
            mood = "Aggressive & Confident"
        elif win_rate < 0.4:
            mood = "Cautious & Defensive"
        else:
            mood = "Neutral & Analytical"
            
        # Determine Market Sentiment (Combine indicators with Macro)
        macro_vibe = macro.get('vibe', 'Neutral') if macro else "Unknown"
        
        if volatility > 0.04 or "Crisis" in macro_vibe:
            sentiment = f"Turbulent / {macro_vibe}"
        elif trend > 0.01:
            sentiment = f"Strongly Bullish ({macro_vibe})"
        elif trend < -0.01:
            sentiment = f"Strongly Bearish ({macro_vibe})"
        else:
            sentiment = f"Consolidating ({macro_vibe})"
            
        observations = [
            f"Average market volatility is at {volatility:.2%}.",
            f"Current win rate stands at {win_rate:.0%}.",
            f"Global Economic Vibe: {macro_vibe}."
        ]
        
        if macro and 'metrics' in macro:
            vix = macro['metrics'].get('VIX', 0)
            observations.append(f"VIX is trading at {vix:.2f}.")

        # Add News context to journal
        if news_agg:
            observations.append(f"I am hearing these key headlines: {', '.join(news_agg[:3])}...")

        if win_rate < 0.3:
            observations.append("I'm struggling to find predictable patterns in this market.")
        elif win_rate > 0.7:
            observations.append("Market conditions are aligning perfectly with my current weights.")
            
        return JournalEntry(
            sentiment=sentiment,
            mood=mood,
            observations=observations
        )
