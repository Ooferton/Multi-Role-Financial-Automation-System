#!/bin/bash

# 0. Ensure binary exists (Download at runtime if needed to bypass build limits)
if [ ! -f "./ollama" ]; then
    echo "Downloading Ollama Service Binary (v0.1.32 Standalone)..."
    curl -L https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64 -o ./ollama
    chmod +x ./ollama
fi

# 1. Start Ollama in the background
echo "Starting Ollama Service (AI Brain)..."
./ollama serve > logs/ollama.log 2>&1 &

# 2. Wait for Ollama to be ready
echo "Waiting for Ollama (127.0.0.1:11434)..."
MAX_ATTEMPTS=30
ATTEMPT=0
while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "Ollama failed to start after $MAX_ATTEMPTS attempts."
        echo "--- OLLAMA ERROR LOG ---"
        cat logs/ollama.log
        exit 1
    fi
    sleep 2
done
echo "Ollama is ONLINE."

# 3. Run the Master Cloud Orchestrator
echo "Starting Unified Cloud Orchestrator..."
python cloud_orchestrator.py
