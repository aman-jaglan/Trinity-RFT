#!/bin/bash
#
# Complete loan failure learning pipeline
# Run this script from Trinity-RFT/examples/loan_failure_learning/
#

set -e  # Exit on any error

echo "🚀 Trinity-RFT Loan Failure Learning Pipeline"
echo "=============================================="

# Check we're in the right directory
if [[ ! -f "loan_failure.yaml" ]]; then
    echo "❌ Error: Run this script from examples/loan_failure_learning/"
    exit 1
fi

# Step 1: Prepare data
echo ""
echo "📂 Step 1: Preparing training data..."
python prepare_data.py

if [[ ! -f "data/loan_failures_train.jsonl" ]]; then
    echo "❌ Error: Data preparation failed"
    exit 1
fi

echo "✅ Data preparation complete"

# Step 2: Check Ray cluster
echo ""
echo "🔍 Step 2: Checking Ray cluster..."

if ! ray status >/dev/null 2>&1; then
    echo "⚠️  Ray cluster not running. Starting..."
    ray start --head
    sleep 5
    
    if ! ray status >/dev/null 2>&1; then
        echo "❌ Error: Failed to start Ray cluster"
        exit 1
    fi
fi

echo "✅ Ray cluster ready"

# Step 3: Run training
echo ""
echo "🎯 Step 3: Starting Trinity-RFT training..."
echo "Monitor progress at: http://localhost:8265 (Ray Dashboard)"
echo ""

# Run Trinity-RFT training with plugin directory
trinity run --config loan_failure.yaml --plugin-dir "$(pwd)"

echo ""
echo "🎉 Training complete!"
echo ""
echo "🔄 To extend to real HuggingFace data:"
echo "   • Modify load_example_data() in prepare_data.py"
echo "   • Replace with: load_dataset('your-username/dataset')"
echo "   • Re-run this script"