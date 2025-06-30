#!/usr/bin/env python3
"""
Explore the Hugging Face dataset schema for loan underwriting data.
"""

from datasets import load_dataset
import json

def explore_dataset():
    """Load and explore the Hugging Face dataset schema"""
    
    print("🔍 Loading Hugging Face dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("Jarrodbarnes/arc-loan-underwriting-trinity-rft")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   Splits: {list(ds.keys())}")
        
        for split_name, split_data in ds.items():
            print(f"\n📋 Split '{split_name}':")
            print(f"   Size: {len(split_data)} samples")
            print(f"   Features: {list(split_data.features.keys())}")
            print(f"   Feature types: {split_data.features}")
            
            # Show first example
            if len(split_data) > 0:
                first_example = split_data[0]
                print(f"\n🔍 First example structure:")
                print(json.dumps(first_example, indent=2, default=str))
                
                # If multi-turn, analyze turn structure
                if 'turns' in first_example:
                    turns = first_example['turns']
                    print(f"\n🔄 Multi-turn analysis:")
                    print(f"   Number of turns: {len(turns)}")
                    if len(turns) > 0:
                        print(f"   Turn structure: {list(turns[0].keys())}")
                        print(f"   First turn example:")
                        print(json.dumps(turns[0], indent=4, default=str))
            
            # Show a few more examples for variety
            print(f"\n📝 Sample IDs (first 5):")
            for i in range(min(5, len(split_data))):
                sample = split_data[i]
                sample_id = sample.get('conversation_id', sample.get('id', f'sample_{i}'))
                print(f"   {i}: {sample_id}")
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

if __name__ == "__main__":
    explore_dataset()