#!/bin/bash

# Create directories
mkdir -p data logs ml/models

# Run the Master Cloud Orchestrator
echo "Starting Unified Cloud Orchestrator..."
python cloud_orchestrator.py
