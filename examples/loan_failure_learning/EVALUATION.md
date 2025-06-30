# Loan Failure Learning - Evaluation Guide

## Trinity Native Evaluation System

This project uses Trinity-RFT's built-in evaluation infrastructure instead of standalone scripts for seamless integration with the training pipeline.

## Evaluation Configuration

### Automatic Evaluation During Training

Evaluation is configured in `loan_failure.yaml`:

```yaml
buffer:
  explorer_input:
    eval_tasksets:
    - name: loan_failures_eval
      storage_type: file  
      path: 'examples/loan_failure_learning/data/loan_failures_test.jsonl'
      format:
        prompt_key: 'user_input'
        response_key: 'failed_response'
      reward_fn: 'loan_failure_reward'
      workflow_type: 'loan_failure_workflow'

explorer:
  eval_interval: 10  # Evaluate every 10 training steps
```

### Evaluation Metrics

The enhanced `LoanFailureReward` provides detailed metrics during evaluation:

- **final_reward**: Binary (0 or 1) - Core training signal
- **sample_weight**: 1.0-2.0 - Training importance based on business priority
- **avoided_failure**: 0-1 - How well the response avoids the specific failure mode
- **improvement_quality**: 0-1 - Overall response quality assessment
- **business_impact_cost**: USD - Financial impact of the failure
- **response_length**: Characters - Response verbosity
- **mcp_server_calls**: Count - Complexity of the original interaction

## Running Evaluation

### 1. Continuous Evaluation (Recommended)

Evaluation runs automatically during training:

```bash
# Start training with automatic evaluation
ray start --head
trinity run --config examples/loan_failure_learning/loan_failure.yaml
```

Metrics are automatically logged to W&B dashboard.

### 2. Benchmark Mode

Evaluate specific checkpoints without training:

```bash
# Evaluate latest checkpoint
trinity bench --config examples/loan_failure_learning/loan_failure.yaml

# Evaluate specific checkpoint
trinity bench --config examples/loan_failure_learning/loan_failure.yaml \
  --checkpoint-path ./checkpoints/loan-failure-grpo/step-100
```

### 3. Multi-Checkpoint Evaluation

Evaluate all saved checkpoints:

```bash
# Evaluate all checkpoints in the directory
trinity bench --config examples/loan_failure_learning/loan_failure.yaml \
  --checkpoint-path ./checkpoints/loan-failure-grpo/ \
  --eval-all
```

## Monitoring Results

### W&B Dashboard

Evaluation metrics are automatically logged to Weights & Biases:

- Navigate to your W&B project: `loan-failure-learning`
- View evaluation metrics in real-time during training
- Compare different checkpoints and training runs

### Key Metrics to Monitor

1. **final_reward**: Should increase as model learns to avoid failures
2. **avoided_failure**: Should approach 1.0 for well-learned failure modes
3. **improvement_quality**: Should increase as responses become more professional
4. **sample_weight distribution**: Shows which failure types are prioritized

## Evaluation Analysis

### Failure Mode Performance

Monitor how well the model learns different failure types:

- **discriminatory_language**: Highest priority - should learn quickly
- **inaccurate_rates**: Medium priority - moderate learning curve  
- **missing_disclosures**: Variable priority - based on business impact
- **inappropriate_tone**: Low priority - may take longer to learn
- **insufficient_information**: Context-dependent priority

### Business Impact Tracking

The evaluation system tracks:

- **High-cost failures**: Priority score 0.8+ (sample weight 1.8+)
- **Medium-cost failures**: Priority score 0.4-0.8 (sample weight 1.4-1.8)
- **Low-cost failures**: Priority score 0.0-0.4 (sample weight 1.0-1.4)

## Advantages Over Standalone Evaluation

✅ **Seamless Integration**: No separate model loading or prompt recreation  
✅ **Distributed Evaluation**: Leverages Ray for scalable evaluation  
✅ **Continuous Monitoring**: Real-time insights during training  
✅ **Checkpoint Comparison**: Easy comparison across training stages  
✅ **Rich Metrics**: Detailed failure analysis and quality assessment  
✅ **Business Alignment**: Priority-based sample weighting for business impact  

## Troubleshooting

### Common Issues

1. **No evaluation metrics in W&B**: Check `eval_interval` setting
2. **Evaluation takes too long**: Reduce test dataset size or increase `eval_interval`
3. **Memory issues during evaluation**: Reduce batch size in evaluation config

### Debug Commands

```bash
# Check evaluation configuration
trinity config validate examples/loan_failure_learning/loan_failure.yaml

# Verify test data format
head -n 1 examples/loan_failure_learning/data/loan_failures_test.jsonl | python -m json.tool

# Test reward function manually
python -c "
from loan_rewards import LoanFailureReward
reward_fn = LoanFailureReward()
print(reward_fn('test response', truth='{\"episode_reward\": 1, \"priority_score\": 0.8}', return_dict=True))
"
```