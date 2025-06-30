"""
Loan Failure Learning Reward Functions

Reward function that uses pre-labeled HuggingFace data with Trinity infrastructure.
"""

import json
from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn


@REWARD_FUNCTIONS.register_module("loan_underwriting_reward")
class LoanUnderwritingReward(RewardFn):
    """Reward function using pre-labeled HuggingFace data + Trinity infrastructure"""
    
    def __call__(self, response, prompt=None, truth=None, return_dict=False):
        """Calculate binary reward using granular reward_breakdown"""
        
        if not truth:
            return 0.0
        
        # Parse pre-labeled data from HuggingFace (passed via Trinity's truth parameter)
        try:
            truth_data = json.loads(truth) if isinstance(truth, str) else truth
        except:
            return 0.0
        
        if not isinstance(truth_data, dict):
            return 0.0
        
        # Use reward_breakdown from dataset
        breakdown = truth_data.get('reward_breakdown', {})
        
        # Binary reward: 1.0 if ALL components pass, 0.0 if ANY fails
        all_passed = all(breakdown.values()) if breakdown else False
        final_reward = 1.0 if all_passed else 0.0
        
        # Sample weighting for training focus
        failed_components = [k for k, v in breakdown.items() if not v]
        sample_weight = 1.0 + len(failed_components) * 0.3  # More weight for failure cases
        
        if return_dict:
            return {
                "final_reward": final_reward,
                "sample_weight": sample_weight, 
                "failed_components": failed_components,
                "breakdown": breakdown,
                "response_length": len(response),
                "business_impact_cost": truth_data.get('business_impact', {}).get('estimated_cost_usd', 0),
                "mcp_server_calls": len(truth_data.get('mcp_server_calls', []))
            }
        
        return final_reward