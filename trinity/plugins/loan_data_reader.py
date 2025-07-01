"""
Custom data reader for loan underwriting dataset with multiple responses per prompt
"""

import json
import logging
from typing import List, Optional, Dict, Any

from trinity.buffer.reader.file_reader import FILE_READERS, RolloutDataReader, _HFBatchReader
from trinity.common.constants import ReadStrategy, TaskType
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.rewards import REWARD_FUNCTIONS
from datasets import load_dataset

logger = logging.getLogger(__name__)


@FILE_READERS.register_module("rollout", force=True)
class LoanUnderwritingDataReader(RolloutDataReader):
    """
    Custom reader for loan underwriting dataset that handles multiple responses per prompt.
    
    This reader:
    1. Loads data from HuggingFace with multiple responses per prompt
    2. Selects the best response based on reward for training
    3. Properly integrates with Trinity's Task and Workflow system
    4. Works with our custom loan_underwriting_reward function
    """
    
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        # Call parent init but we'll override the dataset loading
        self.meta = meta
        self.split = meta.split
        subset_name = meta.subset_name
        
        # Load the dataset
        raw_dataset = load_dataset(
            meta.path, 
            name=subset_name, 
            split=self.split, 
            trust_remote_code=True
        )
        
        # Preprocess to select best responses
        processed_dataset = self._preprocess_dataset(raw_dataset)
        
        # Create batch reader with processed dataset
        self.dataset = _HFBatchReader(
            processed_dataset,
            name=meta.name,
            max_epoch=meta.total_epochs if meta.task_type == TaskType.EXPLORE else 1,
            offset=meta.index,
            drop_last=meta.task_type == TaskType.EXPLORE,
        )
        
        self.read_batch_size = config.batch_size
        self.prompt_key = meta.format.prompt_key
        self.response_key = meta.format.response_key
        self.workflow_key = meta.format.workflow_key if hasattr(meta.format, 'workflow_key') else None
        self.reward_fn_key = meta.format.reward_fn_key if hasattr(meta.format, 'reward_fn_key') else None
        
        self.task_type = meta.task_type
        self.default_workflow_cls = WORKFLOWS.get(meta.default_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(meta.default_reward_fn_type)
        
        logger.info(f"Initialized LoanUnderwritingDataReader for {meta.path} with {len(processed_dataset)} samples")
    
    def _preprocess_dataset(self, dataset):
        """
        Preprocess the dataset to handle multiple responses format.
        Selects the best response based on reward for training.
        """
        processed_data = []
        
        for sample in dataset:
            # Extract prompt
            prompt = sample[self.prompt_key]
            
            # Handle multiple responses
            if self.response_key in sample and isinstance(sample[self.response_key], list):
                responses = sample[self.response_key]
                
                if responses:
                    # Select best response based on reward
                    best_response = max(responses, key=lambda r: r.get('reward', 0))
                    
                    # Create processed sample with selected response
                    processed_sample = {
                        self.prompt_key: prompt,
                        'response': best_response['response'],  # Single response
                        'reward': best_response['reward'],
                        'metadata': sample.get('metadata', {}),
                        'response_metadata': best_response.get('metadata', {}),
                        # Keep original responses for reference if needed
                        'all_responses': responses
                    }
                else:
                    # No responses, use empty
                    processed_sample = {
                        self.prompt_key: prompt,
                        'response': '',
                        'reward': 0.0,
                        'metadata': sample.get('metadata', {}),
                        'all_responses': []
                    }
            else:
                # Single response format (fallback)
                processed_sample = {
                    self.prompt_key: prompt,
                    'response': sample.get(self.response_key, ''),
                    'reward': sample.get('reward', 0.0),
                    'metadata': sample.get('metadata', {}),
                    'all_responses': []
                }
            
            processed_data.append(processed_sample)
        
        # Convert back to dataset format
        from datasets import Dataset
        return Dataset.from_list(processed_data)
    
    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List[Task]:
        """
        Read batch of tasks with proper format for Trinity framework.
        
        Returns:
            List of Task objects ready for workflow processing
        """
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples = self.dataset.read_batch(batch_size)
        
        for sample in samples:
            # Get workflow class (use default since our dataset doesn't have workflow key)
            workflow_class = self.default_workflow_cls
            
            # Get reward function (use our custom loan underwriting reward)
            reward_fn = self.default_reward_fn_cls
            
            assert workflow_class is not None, "`default_workflow_type` must be specified"
            
            # Create task with properly formatted data
            # The raw_task will be used by the workflow to access prompt and response
            raw_task = {
                'prompt': sample[self.prompt_key],
                'response': sample['response'],  # Already selected best response
                'reward': sample.get('reward', 0.0),
                'metadata': sample.get('metadata', {}),
                'response_metadata': sample.get('response_metadata', {}),
                # Optional: include all responses for analysis
                'all_responses': sample.get('all_responses', [])
            }
            
            task = Task(
                workflow=workflow_class,
                format_args=self.meta.format,
                rollout_args=self.meta.rollout_args,
                workflow_args=self.meta.workflow_args if hasattr(self.meta, 'workflow_args') else None,
                is_eval=self.meta.task_type == TaskType.EVAL,
                reward_fn=reward_fn,
                raw_task=raw_task,
            )
            tasks.append(task)
        
        logger.debug(f"Read {len(tasks)} tasks from {len(samples)} samples")
        return tasks