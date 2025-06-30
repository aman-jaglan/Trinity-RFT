#!/usr/bin/env python3
"""
Prepare loan failure data using Trinity-RFT's built-in data processing system.

This script uses Trinity-RFT's professional data pipeline instead of custom preprocessing.
"""

from pathlib import Path
import sys
import json
from datasets import load_dataset

from trinity.common.config import BufferConfig, DataPipelineConfig, StorageConfig, FormatConfig
from trinity.common.constants import StorageType
from trinity.data.core.dataset import RftDataset
from trinity.data.core.formatter import BaseDataFormatter


class LoanUnderwritingFormatter(BaseDataFormatter):
    """Custom formatter for loan underwriting data using Trinity-RFT's system"""
    
    def transform(self, sample: dict) -> dict:
        """Transform HF sample to Trinity-RFT format (keeping multi-response structure)"""
        
        # Parse the prompt JSON to get application ID
        prompt_data = json.loads(sample['prompt'])
        application_id = prompt_data['loan_application']['application_id']
        
        # Keep the multi-response structure but add Trinity-RFT fields
        transformed = {
            # Trinity-RFT required fields
            'id': application_id,
            'task_type': 'loan_underwriting',
            
            # Core fields for training - use first response as primary
            'prompt': sample['prompt'],  # Full loan application as prompt
            'response': sample['responses'][0]['response'],  # First response as primary
            'reward': sample['responses'][0]['reward'],
            
            # Keep all responses for multi-response training
            'all_responses': sample['responses'],
            
            # Metadata
            'application_id': application_id,
            'challenge_types': sample['metadata']['challenge_types'],
            'num_responses': len(sample['responses']),
            
            # Keep original for reference
            'raw_sample': sample
        }
        
        return transformed


def prepare_data_with_trinity():
    """Prepare data using Trinity-RFT's data processing system with Hugging Face dataset"""
    
    print("🚀 Loading loan underwriting data from Hugging Face...")
    
    # Load dataset from Hugging Face
    try:
        hf_dataset = load_dataset("Jarrodbarnes/arc-loan-underwriting-trinity-rft-v2")
        print(f"✅ Loaded HF dataset with {len(hf_dataset['train'])} samples")
    except Exception as e:
        print(f"❌ Error loading HF dataset: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Convert HF dataset to Trinity-RFT format
    print("🔧 Converting to Trinity-RFT format...")
    
    formatter_config = FormatConfig(
        prompt_key='prompt',
        response_key='response'
    )
    
    # Configure Trinity-RFT data pipeline to use the original HF dataset directly
    data_pipeline_config = DataPipelineConfig(
        input_buffers=[
            StorageConfig(
                name="loan_underwriting_input",
                storage_type=StorageType.FILE,
                path="Jarrodbarnes/arc-loan-underwriting-trinity-rft-v2",  
                raw=True
            )
        ],
        output_buffer=StorageConfig(
            name="loan_underwriting_output",
            storage_type=StorageType.FILE,
            path=str(output_dir / "loan_underwriting_processed.jsonl")
        )
    )
    
    buffer_config = BufferConfig()
    
    # Initialize Trinity-RFT dataset
    print("📊 Loading through Trinity-RFT system...")
    dataset = RftDataset(
        data_pipeline_config=data_pipeline_config,
        buffer_config=buffer_config,
        track_lineage=True
    )
    
    # Read from input buffer  
    dataset.read_from_buffer()
    print(f"📥 Loaded {len(dataset.data)} samples")
    
    # Apply our custom formatter
    print("🔧 Applying loan underwriting formatter...")
    loan_formatter = LoanUnderwritingFormatter(formatter_config)
    dataset.format([loan_formatter])
    
    print(f"✅ Formatted {len(dataset.data)} loan applications")
    
    # Split into train/test using Trinity-RFT's built-in methods
    total_size = len(dataset.data)
    train_size = int(total_size * 0.8)
    
    # Create train set
    train_dataset = RftDataset(data_pipeline_config, buffer_config)
    train_dataset.data = dataset.data.select(range(train_size))
    
    # Create test set  
    test_dataset = RftDataset(data_pipeline_config, buffer_config)
    test_dataset.data = dataset.data.select(range(train_size, total_size))
    
    print(f"📂 Split: {len(train_dataset.data)} train, {len(test_dataset.data)} test")
    
    # Write to output buffers using Trinity-RFT's system
    train_config = StorageConfig(
        name="loan_underwriting_train",
        storage_type=StorageType.FILE, 
        path=str(output_dir / "loan_underwriting_train.jsonl")
    )
    
    test_config = StorageConfig(
        name="loan_underwriting_test",
        storage_type=StorageType.FILE,
        path=str(output_dir / "loan_underwriting_test.jsonl") 
    )
    
    train_dataset.write_to_buffer(train_config, buffer_config)
    test_dataset.write_to_buffer(test_config, buffer_config)
    
    print(f"💾 Saved using Trinity-RFT data engine:")
    print(f"   Training: {output_dir / 'loan_underwriting_train.jsonl'}")
    print(f"   Testing: {output_dir / 'loan_underwriting_test.jsonl'}")
    
    # No temp files to clean up
    
    # Analysis using Trinity-RFT capabilities
    analyze_with_trinity(train_dataset)
    
    print(f"\n🎯 Ready for Trinity-RFT training!")
    print(f"   Next: ray start --head")
    print(f"   Then: trinity run --config loan_underwriting.yaml")


def analyze_with_trinity(dataset: RftDataset):
    """Analyze loan underwriting data using Trinity-RFT's built-in capabilities"""
    
    print(f"\n📊 Data Analysis (via Trinity-RFT):")
    print(f"   Total training examples: {len(dataset.data)}")
    
    # Analyze the loan underwriting dataset
    decisions = {}
    rewards = []
    challenge_types = {}
    failure_counts = {'has_failure': 0, 'no_failure': 0}
    synthetic_counts = {'synthetic': 0, 'real': 0}
    
    for item in dataset.data:
        # Parse response to get decision
        try:
            response_data = json.loads(item.get('response', '{}'))
            decision = response_data.get('decision', 'UNKNOWN')
        except:
            decision = 'UNKNOWN'
        decisions[decision] = decisions.get(decision, 0) + 1
        
        # Collect rewards
        rewards.append(item.get('reward', 0))
        
        # Analyze challenge types
        challenges = item.get('challenge_types', [])
        for challenge in challenges:
            challenge_types[challenge] = challenge_types.get(challenge, 0) + 1
        
        # Count failures from all_responses
        all_responses = item.get('all_responses', [])
        has_any_failure = any(resp.get('metadata', {}).get('has_failure', False) for resp in all_responses)
        if has_any_failure:
            failure_counts['has_failure'] += 1
        else:
            failure_counts['no_failure'] += 1
        
        # Count synthetic vs real from all_responses
        has_synthetic = any(resp.get('metadata', {}).get('synthetic', False) for resp in all_responses)
        if has_synthetic:
            synthetic_counts['synthetic'] += 1
        else:
            synthetic_counts['real'] += 1
    
    # Report results
    print(f"   Final decisions:")
    for decision, count in sorted(decisions.items(), key=lambda x: x[1], reverse=True):
        print(f"     {decision}: {count}")
    
    print(f"   Challenge types:")
    for challenge, count in sorted(challenge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"     {challenge}: {count}")
    
    print(f"   Failure distribution:")
    print(f"     Has failure: {failure_counts['has_failure']}")
    print(f"     No failure: {failure_counts['no_failure']}")
    
    print(f"   Data source:")
    print(f"     Synthetic: {synthetic_counts['synthetic']}")
    print(f"     Real: {synthetic_counts['real']}")
    
    if rewards:
        print(f"   Reward distribution:")
        print(f"     Mean: {sum(rewards)/len(rewards):.3f}")
        print(f"     Range: {min(rewards):.3f} - {max(rewards):.3f}")
        print(f"     Count by range:")
        low_rewards = sum(1 for r in rewards if r < 2.0)
        mid_rewards = sum(1 for r in rewards if 2.0 <= r < 4.0)
        high_rewards = sum(1 for r in rewards if r >= 4.0)
        print(f"       Low (< 2.0): {low_rewards}")
        print(f"       Mid (2.0-4.0): {mid_rewards}")
        print(f"       High (>= 4.0): {high_rewards}")


def main():
    """Main entry point using Trinity-RFT's data processing"""
    prepare_data_with_trinity()


if __name__ == "__main__":
    main()