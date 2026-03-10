import time
import os
import logging
import json
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from agents.trading_agent import TradingAgent
from agents.alpaca_broker import AlpacaBroker
from agents.coinbase_broker import CoinbaseBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from strategies.swing import SwingStrategy
from strategies.hft import HFTStrategy
from strategies.day_trading import DayTradingStrategy
from core.orchestrator import Orchestrator
from data.feature_store import FeatureStore, MarketTick
from agents.mock_broker import MockBroker
from ml.quant_models import estimate_regime, calculate_risk_parity_weights
from data.searxng_search import SearXNGClient
import pytz
import argparse

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveRunner")

def run_warmup(broker, agent_swarm, feature_store, target_symbols_dict):
    """Fetches historical data to prime the agent and dashboard."""
    logger.info("Starting Historical Warmup (Last 100 bars within 72-hour window)...")
    
    # Use a 3-day lookback to ensure we catch the last trading day even on weekends
    start = datetime.now() - timedelta(days=3)
    end = datetime.now()
    
    total_bars = 0
    for sector, symbols in target_symbols_dict.items():
        for symbol in symbols:
            # Emergency Circuit Breaker Check (Inside Warmup)
            if os.path.exists("data/circuit_breaker.lock"):
                logger.critical(f"[CIRCUIT BREAKER] HALTING WARMUP for {symbol}!")
                while os.path.exists("data/circuit_breaker.lock"):
                    time.sleep(2)
                logger.info("[CIRCUIT BREAKER] Reset detected. Resuming warmup...")
    
            try:
                # Fetch last 100 bars from the window
                bars = broker.get_historical_data(symbol, start=start, end=end, timeframe='1Min', limit=100)
                if bars:
                    for bar in bars:
                        # Save OHLC as separate ticks so dashboard resample can reconstruct candles
                        # Use 1-second offsets to ensure uniqueness in SQLite PK (symbol, timestamp)
                        for i, attr in enumerate(['open', 'high', 'low', 'close']):
                            # Ensure bar.timestamp is UTC
                            base_ts = bar.timestamp
                            if base_ts.tzinfo is None:
                                base_ts = pytz.UTC.localize(base_ts)
                                
                            tick = MarketTick(
                                symbol=symbol,
                                timestamp=base_ts + timedelta(seconds=i),
                                price=getattr(bar, attr),
                                size=bar.volume / 4.0,
                                exchange="WARMUP"
                            )
                            if attr == 'close':
                                agent_swarm[sector].update_market_state({'tick': tick})
                            feature_store.save_tick(tick)
                    total_bars += len(bars)
            except Exception as e:
                logger.error(f"Warmup failed for {symbol}: {e}")
            
    logger.info(f"Warmup complete. Ingested {total_bars} historical data points across sectors.")

def start_dashboard():
    """Launches the Streamlit dashboard in the background."""
    logger.info("Launching Sentience Dashboard (http://localhost:8501)...")
    try:
        # Create logs directory if missing
        os.makedirs("logs", exist_ok=True)
        
        # Open a log file for the dashboard
        log_f = open("logs/dashboard.log", "w")
        
        # Using sys.executable to ensure we use the same environment
        # Removed DEVNULL redirect to allow errors to go to dashboard.log
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true"],
            stdout=log_f,
            stderr=log_f,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=True if os.name != 'nt' else False
        )
        logger.info("✅ Dashboard process started in background.")
        logger.info("💡 If the browser doesn't open, visit: http://localhost:8501")
        logger.info("⚠️ To run manually if background fails: 'python -m streamlit run dashboard.py'")
    except Exception as e:
        logger.error(f"❌ Failed to launch dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Role Financial Trading Runner")
    parser.add_argument("--one-shot", action="store_true", help="Run once and exit (for GitHub Actions)")
    parser.add_argument("--crypto-etf-only", action="store_true", help="Only trade BTC and Bitcoin/Market ETFs (Bypasses PDT)")
    parser.add_argument("--skip-dashboard", action="store_true", help="Do not auto-launch the dashboard (for HF Spaces)")
    args = parser.parse_args()

    load_dotenv()
    
    # 0. Process ID Tracking
    os.makedirs("data", exist_ok=True)
    
    # 0.1 Auto-Reset Circuit Breaker on Startup
    if os.path.exists("data/circuit_breaker.lock"):
        try:
            os.remove("data/circuit_breaker.lock")
            logger.info("♻️ System Startup: Circuit Breaker auto-reset.")
        except: pass

    with open("data/live_runner.pid", "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"Process ID: {os.getpid()}")

    # 1. Initialize Components
    logger.info("Initializing Live Trading System...")
    
    # Auto-launch Dashboard (only if not in one-shot mode and not skipped)
    if not args.one_shot and not args.skip_dashboard:
        start_dashboard()
    
    # Orchestrator with config
    orchestrator = Orchestrator("config/config.yaml")

    # 1. Initialize Broker based on Config
    broker_name = orchestrator.config.get('brokerage', {}).get('name', 'MOCK')
    
    if broker_name == "ALPACA" and os.getenv("APCA_API_KEY_ID"):
        logger.info("Initializing Alpaca Broker...")
        broker = AlpacaBroker()
    elif broker_name == "COINBASE":
        logger.info("Initializing Coinbase Broker (Crypto)...")
        broker = CoinbaseBroker()
    else:
        logger.info("Initializing Mock Broker for simulation...")
        broker = MockBroker(initial_cash=100000)
    
    # Connect Orchestrator
    orchestrator.set_broker(broker)
    
    # Initial PDT Check
    try:
        if hasattr(broker, 'get_account_summary'):
            acc_info = broker.get_account_summary()
            if acc_info:
                logger.info(f"Initial Account Load | Equity: ${acc_info.get('equity')} | PDT Status: {acc_info.get('pattern_day_trader')} | Day Trades: {acc_info.get('day_trade_count')}")
                if acc_info.get('equity', 0) < 25000 and acc_info.get('day_trade_count', 0) >= 3:
                    logger.warning("PDT LIMIT APPROACHED! System will throttle high-frequency trades to avoid lock-out.")
    except Exception as e:
        logger.error(f"Failed to perform initial PDT check: {e}")
    
    # Feature Store (for logging history if needed)
    feature_store = FeatureStore(db_path="data/feature_store.db")
    
    # --- TARGET UNIVERSE CONFIGURATION ---
    # Default fallback symbols
    target_symbols_dict = {
        "RESEARCH_PICKS": [], # Populated dynamically below
        "INDEX": ["SPY", "QQQ", "DIA", "IWM"],
        "TECH": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
        "FINANCE": ["BAC", "JPM", "GS", "V", "XLF"],
        "BONDS": ["TLT", "IEF", "BND"],
        "CRYPTO": ["BTC/USD", "ETH/USD"]
    }

    # Load Dynamic Targets from Autonomous Researcher
    targets_path = "data/daily_targets.json"
    if os.path.exists(targets_path):
        try:
            with open(targets_path, "r") as f:
                daily_data = json.load(f)
                dynamic_list = daily_data.get("dynamic_targets", [])
                if dynamic_list:
                    logger.info(f"🚀 Research Lab Injection: Using {len(dynamic_list)} high-conviction targets: {dynamic_list}")
                    target_symbols_dict["RESEARCH_PICKS"] = dynamic_list
                    # Optional: In strict research mode, we could clear other sectors
                    # target_symbols_dict = {"RESEARCH_PICKS": dynamic_list}
        except Exception as e:
            logger.error(f"Failed to load dynamic targets: {e}")
    
    flat_symbols = [symbol for sector in target_symbols_dict.values() for symbol in sector]
    
    # Swarm Agent Initialization
    agent_swarm = {
        sector: TradingAgent(f"Live_Swarm_{sector}", {}, feature_store, broker) 
        for sector in target_symbols_dict.keys()
    }
    
    # --- DYNAMIC STRATEGY LOADING ---
    STRATEGY_CONFIG_PATH = "data/active_strategies.json"
    
    def get_market_data_for_regime_and_parity(target_symbol="SPY"):
        try:
            logger.info(f"📊 Calculating Market Regime & Risk Parity...")
            # Fetch 100 days of daily data for SPY (Regime + CVaR) and QQQ, TLT (for parity example)
            start = datetime.now() - timedelta(days=150)
            bars = broker.get_historical_data(target_symbol, start=start, end=datetime.now(), timeframe='1Day', limit=100)
            if not bars: return 'UNKNOWN', 'NEUTRAL', None
            
            closes = [b.close for b in bars]
            returns = pd.Series(closes).pct_change().dropna().values
            regime = estimate_regime(returns)
            
            # Augment with Live News Sentiment
            search_client = SearXNGClient()
            macro_sentiment = search_client.get_macro_sentiment()
            
            # If news is highly bearish but chart says bull, we might want to downgrade to choppy
            if macro_sentiment == "BEARISH_MACRO" and regime == "BULL":
                logger.warning("Brave Search detected heavy BEARISH news. Downgrading BULL regime to CHOPPY.")
                regime = "CHOPPY"
            elif macro_sentiment == "BULLISH_MACRO" and regime in ["CHOPPY", "BEAR"]:
                logger.info("Brave Search detected strong BULLISH news. Market fundamentals improving.")
            
            
            # Send SPY returns to Risk Manager for Proxy Market CVaR
            orchestrator.risk_manager.check_portfolio_risk(returns.tolist())
            
            logger.info(f"Current Market Regime: {regime}")
            return regime, macro_sentiment, returns
        except Exception as e:
            logger.error(f"Failed to estimate regime: {e}")
            return 'UNKNOWN', 'NEUTRAL', None
            
    def update_risk_parity(broker, target_symbols, orchestrator):
        try:
            logger.info("⚖️ Calculating Risk Parity (Inverse Volatility) Caps...")
            # We take a sample basket (Top 10) to save API calls in testing
            subset = target_symbols[:10]
            returns_dict = {}
            start = datetime.now() - timedelta(days=60)
            
            for sym in subset:
                bars = broker.get_historical_data(sym, start=start, end=datetime.now(), timeframe='1Day', limit=60)
                if bars:
                    returns_dict[sym] = [b.close for b in bars]
            
            if not returns_dict: return
            
            # Align lengths
            min_len = min([len(v) for v in returns_dict.values()])
            if min_len < 10: return
            
            df = pd.DataFrame({k: pd.Series(v[-min_len:]).pct_change().dropna().values for k, v in returns_dict.items()})
            weights = calculate_risk_parity_weights(df)
            orchestrator.risk_manager.set_parity_weights(weights)
        except Exception as e:
            logger.error(f"Failed to update risk parity: {e}")
            
    def update_swarm_allocations(broker, target_symbols_dict, orchestrator):
        try:
            logger.info("🧠 Meta-Orchestrator: Calculating Swarm Capital Allocations (Sharpe-based)...")
            allocations = {}
            start = datetime.now() - timedelta(days=60)
            
            for sector, symbols in target_symbols_dict.items():
                if not symbols: continue
                # Use first symbol as proxy for sector performance
                proxy = symbols[-1] if 'XL' in symbols[-1] else symbols[0] 
                bars = broker.get_historical_data(proxy, start=start, end=datetime.now(), timeframe='1Day', limit=60)
                if not bars: 
                    allocations[sector] = 1.0
                    continue
                
                returns = pd.Series([b.close for b in bars]).pct_change().dropna()
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.5
                
                # Apply high-conviction boost for Research Lab picks
                if sector == "RESEARCH_PICKS":
                    sharpe *= 1.25
                    
                allocations[sector] = max(0.1, sharpe) # Floor at 0.1
                
            # Normalize allocations around 1.0 (mean)
            avg_sharpe = sum(allocations.values()) / len(allocations) if allocations else 1.0
            if avg_sharpe == 0: avg_sharpe = 1.0
            
            normalized_allocations = {s: v / avg_sharpe for s, v in allocations.items()}
            # Cap the multiplier (0.5 to 1.5 multiplier)
            normalized_allocations = {s: max(0.5, min(1.5, v)) for s, v in normalized_allocations.items()}
            
            orchestrator.risk_manager.set_sector_allocations(normalized_allocations)
            logger.info(f"Swarm Allocations Updated: {normalized_allocations}")
            
        except Exception as e:
            logger.error(f"Failed to update Swarm Allocations: {e}")
    
    
    def load_active_strategies(target_swarm, regime='UNKNOWN'):
        # 1. Read dashboard manual override
        mode_path = "data/trading_mode.json"
        trading_mode = "Auto / Dynamic"
        if os.path.exists(mode_path):
            try:
                with open(mode_path, "r") as f:
                    saved = json.load(f)
                    trading_mode = saved.get("mode", "Auto / Dynamic")
            except: pass
            
        logger.info(f"⚙️ Active Trading Mode: {trading_mode} (Regime: {regime})")
        
        # Helper to safely load RL model
        def get_rl_strat(sector):
            model_path = "ml/models/ppo_v3_cyborg"
            if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
                model_path = "ml/models/ppo_trading_real_v2"
            
            stripped_path = model_path.replace(".zip","")
            logger.info(f"🧠 [{sector} Swarm] Loading RL Brain Model: {stripped_path}")
            return RLStrategyV2(f"RL_{sector}", orchestrator.config, broker, stripped_path)
        
        for sector, agent in target_swarm.items():
            agent.clear_strategies()
            
            # --- AUTO / DYNAMIC MODE (Regime-based) ---
            if trading_mode == "Auto / Dynamic":
                if regime in ["BULL"]:
                    agent.add_strategy(get_rl_strat(sector))
                    agent.add_strategy(SwingStrategy(f"Swing_{sector}", {"window_size": 24}, broker))
                else: 
                    agent.add_strategy(get_rl_strat(sector))
            
            # --- DAY TRADING OVERRIDE ---
            elif trading_mode == "Day Trading":
                agent.add_strategy(get_rl_strat(sector))
                agent.add_strategy(DayTradingStrategy(f"DayTrade_{sector}", {"momentum": 1.005}))
                
            # --- HFT OVERRIDE ---
            elif trading_mode == "High Frequency Trading":
                agent.add_strategy(HFTStrategy(f"HFT_{sector}", {"window_size": 50}, broker))
                
            # --- SWING OVERRIDE ---
            elif trading_mode == "Swing Trading":
                agent.add_strategy(SwingStrategy(f"Swing_{sector}", {"window_size": 24}, broker))

    # Initial Load
    current_regime, current_macro, _ = get_market_data_for_regime_and_parity("SPY")
    orchestrator.run_llm_supervision(current_macro, current_regime)
    update_risk_parity(broker, flat_symbols, orchestrator)
    update_swarm_allocations(broker, target_symbols_dict, orchestrator)
    load_active_strategies(agent_swarm, regime=current_regime)
    _last_strat_check = os.path.getmtime(STRATEGY_CONFIG_PATH) if os.path.exists(STRATEGY_CONFIG_PATH) else 0
    _last_regime_check = time.time()
    
    # Broadcast INITIAL state immediately
    try:
        status_path = "data/sentience_status.json"
        with open(status_path, "w") as f:
            json.dump({
                "pid": os.getpid(),
                "mode": "FULL_PORTFOLIO",
                "active": True,
                "timestamp": datetime.now().isoformat()
            }, f)
    except: pass
    
    # 2. Performance Warmup (Historical Data)
    logger.info("Initializing Sectoral Portfolio Swarm...")
    run_warmup(broker, agent_swarm, feature_store, target_symbols_dict)
    
    try:
        while True:
            # Broadcast state to dashboard status JSON
            try:
                status_path = "data/sentience_status.json"
                if os.path.exists(status_path):
                    with open(status_path, "r") as f:
                        curr = json.load(f)
                else: curr = {}
                curr["pid"] = os.getpid()
                curr["mode"] = "FULL_PORTFOLIO"
                curr["active"] = True
                with open(status_path, "w") as f:
                    json.dump(curr, f)
            except: pass

            # 2. Emergency Circuit Breaker Check (Initial)
            if os.path.exists("data/circuit_breaker.lock"):
                logger.critical("[CIRCUIT BREAKER] TRIGGERED! Halting system for safety.")
                while os.path.exists("data/circuit_breaker.lock"): time.sleep(2)
                logger.info("[CIRCUIT BREAKER] Reset detected. Resuming operation...")

            # 2. Check for Strategy Changes
            if os.path.exists(STRATEGY_CONFIG_PATH):
                mtime = os.path.getmtime(STRATEGY_CONFIG_PATH)
                if mtime > _last_strat_check:
                    logger.warning("🔔 Strategy Configuration Change Detected! Reloading Brain...")
                    load_active_strategies(agent, regime=current_regime)
                    _last_strat_check = mtime
                    
            # 2.5 Periodic Regime & Risk Check (Every 4 hours = 14400 secs)
            if time.time() - _last_regime_check > 14400:
                new_regime, recent_macro, _ = get_market_data_for_regime_and_parity("SPY")
                orchestrator.run_llm_supervision(recent_macro, new_regime)
                update_risk_parity(broker, flat_symbols, orchestrator)
                update_swarm_allocations(broker, target_symbols_dict, orchestrator)
                if new_regime != current_regime and new_regime != 'UNKNOWN':
                    logger.warning(f"🚨 MARKET REGIME SHIFT DETECTED: {current_regime} -> {new_regime}")
                    orchestrator.telegram.send_regime_shift(new_regime)
                    current_regime = new_regime
                    load_active_strategies(agent_swarm, regime=current_regime)
                _last_regime_check = time.time()

            # 3. Fetch Latest Prices & Quotes
            prices = broker.get_latest_prices(flat_symbols)
            quotes = broker.get_latest_quotes(flat_symbols)
            
            for sector, symbols in target_symbols_dict.items():
                for symbol in symbols:
                    # Secondary Emergency Check (During long loops)
                    if os.path.exists("data/circuit_breaker.lock"):
                        break
                    
                    price = prices.get(symbol, 0.0)
                    quote = quotes.get(symbol, {})
                    bid_size = quote.get('bid_size', 0.0)
                    ask_size = quote.get('ask_size', 0.0)
                    
                    if price > 0:
                        # 2. Create Tick
                        tick = MarketTick(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            price=price,
                            size=100, # Placeholder volume
                            exchange=broker_name,
                            bid_size=bid_size,
                            ask_size=ask_size
                        )
                        
                        # 3. Update Swarm Agent (Predict & Execute)
                        agent_swarm[sector].update_market_state({'tick': tick})
                        
                        # 3.5 Persist Tick for Dashboard Visualization
                        feature_store.save_tick(tick)
                        
                    else:
                        logger.warning(f"Failed to fetch price for {symbol}.")

            # 4. Exit if in one-shot mode
            if args.one_shot:
                logger.info("One-shot execution complete. Exiting.")
                break

            # 5. Wait (High frequency for Crypto/ETF, 1s for standard)
            sleep_time = 0.1 if args.crypto_etf_only else 1
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Live Trading Stopped by User.")
    except Exception as e:
        logger.exception(f"Crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
