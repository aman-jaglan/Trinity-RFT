# Loan Failure Learning Example

Train loan agents to learn from documented failure trajectories using Trinity-RFT.

## Quick Start

```bash
# 1. Prepare example data
python prepare_data.py

# 2. Start Ray cluster  
ray start --head

# 3. Run training
trinity run --config loan_failure.yaml

# 4. Evaluate (optional)
python evaluate.py --model-path checkpoints/global_step_100
```

## What This Example Demonstrates

- **Custom Workflow**: `LoanFailureWorkflow` for processing failure trajectories
- **Custom Reward**: `LoanFailureReward` using pre-labeled business impact scores  
- **GRPO Training**: Learning to avoid documented failure modes
- **Real Data**: 10 realistic multi-agent loan advisor failures with MCP server calls

## Files

- `loan_failure.yaml` - Main Trinity-RFT configuration
- `train_loan_failure.yaml` - VERL trainer configuration  
- `prepare_data.py` - Convert example trajectories to Trinity-RFT format
- `evaluate.py` - Test trained model improvements
- `loan_workflows.py` - Custom workflow plugin
- `loan_rewards.py` - Custom reward plugin

## Extending to Real Data

Replace example data in `prepare_data.py` with your HuggingFace dataset:

```python
# Change this line:
dataset = load_local_examples()

# To this:
dataset = load_dataset("your-username/loan-failure-dataset")
```

The rest of the pipeline remains the same.