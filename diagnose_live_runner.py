import time
import os
import yaml
from datetime import datetime
from agents.alpaca_broker import AlpacaBroker
from agents.mock_broker import MockBroker
from strategies.rl_strategy_v2 import RLStrategyV2
from data.feature_store import FeatureStore, MarketTick
from ml.reasoning_engine import ReasoningEngine

def run_due_diligence():
    print("Live Runner Deep Due Diligence")
    print("=" * 40)

    # 1. Config & Environment
    print("[1] Environment Audit")
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"  - Config Loaded: {config.get('system', {}).get('mode')}")
    else:
        print("  - [!] config.yaml MISSING")
        return

    # 2. Broker Connection
    print("\n[2] Broker Connectivity")
    broker_name = config.get('brokerage', {}).get('name', 'MOCK')
    if broker_name == "ALPACA":
        if os.getenv("APCA_API_KEY_ID"):
            print("  - Alpaca Keys Found.")
            try:
                broker = AlpacaBroker(paper=True)
                summary = broker.get_account_summary()
                print(f"  - Alpaca Connection: SUCCESS (Equity: ${summary.get('equity', 0)})")
            except Exception as e:
                print(f"  - [!] Alpaca Connection FAILED: {e}")
        else:
            print("  - [!] Alpaca Keys MISSING in .env")
    else:
        print("  - Using Mock Broker.")

    # 3. Strategy & Agents Audit
    print("\n[3] Board of Agents Initialization")
    try:
        # We'll use a dummy broker for strategy testing to avoid live impact
        test_broker = MockBroker()
        model_path = "ml/models/ppo_trading_real_v2"
        strategy = RLStrategyV2("Audit_Strategy", config, test_broker, model_path)
        
        print(f"  - Sentinel: ONLINE")
        print(f"  - Economist: ONLINE (Outlook: {strategy.economist.update_outlook().get('vibe')})")
        print(f"  - News Engine: ONLINE")
        print(f"  - Risk Manager: ONLINE")
        print(f"  - RL Model: LOADED ({model_path})")
    except Exception as e:
        print(f"  - [!] Strategy Init FAILED: {e}")

    # 4. Latency Check
    print("\n[4] Component Latency Audit")
    start = time.time()
    strategy.news_engine.get_sentiment("SPY")
    print(f"  - News Sentiment Latency: {(time.time() - start)*1000:.2f}ms")
    
    start = time.time()
    strategy.economist.update_outlook()
    print(f"  - Economist Outlook Latency: {(time.time() - start)*1000:.2f}ms")

    # 5. Rate Limit Safety
    print("\n[5] Rate Limit Due Diligence")
    # Test caching
    t1 = time.time()
    test_broker.get_account_summary()
    d1 = time.time() - t1
    print(f"  - First API Call: {d1*1000:.2f}ms")
    
    # RLStrategyV2 should cache for 3 seconds
    strategy.broker = AlpacaBroker(paper=True) if broker_name=="ALPACA" else MockBroker()
    strategy.on_tick(MarketTick("SPY", datetime.now(), 500, 100, "EX"))
    t2 = time.time()
    strategy.on_tick(MarketTick("QQQ", datetime.now(), 400, 100, "EX"))
    d2 = time.time() - t2
    print(f"  - Cached API Call (should be fast): {d2*1000:.2f}ms")

    print("\n" + "=" * 40)
    print("Due Diligence Complete.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_due_diligence()
