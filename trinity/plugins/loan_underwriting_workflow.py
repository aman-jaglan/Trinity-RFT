"""
Loan Underwriting Multi-Agent Workflow

This workflow implements a multi-agent loan underwriting system with three agents:
- Loan Officer: Reviews application and employment details
- Credit Analyst: Analyzes credit and calculates risk metrics
- Risk Manager: Makes final approval decision and sets terms
"""

import json
import logging
from typing import List, Dict, Any

from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow
from trinity.common.experience import Experience
from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS

logger = logging.getLogger(__name__)


@WORKFLOWS.register_module("loan_underwriting_workflow")
class LoanUnderwritingWorkflow(SimpleWorkflow):
    """Multi-agent workflow for loan underwriting with verifiable rewards"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the custom reward function if not specified
        if self.reward_fn is None:
            self.reward_fn = REWARD_FUNCTIONS.get("loan_underwriting_reward")()
            logger.info("Using loan_underwriting_reward function")
    
    def format_messages(self) -> List[Dict[str, str]]:
        """Format messages for multi-agent loan underwriting"""
        task_data = self.task.raw_task
        
        # Handle different data formats
        prompt = task_data.get('prompt', '')
        
        # If prompt is a string containing JSON, parse it
        if isinstance(prompt, str):
            try:
                prompt_data = json.loads(prompt)
            except json.JSONDecodeError:
                # If it's not JSON, treat as plain text
                prompt_data = {'loan_application': prompt}
        else:
            prompt_data = prompt
        
        # Extract loan application data
        if isinstance(prompt_data, dict) and 'loan_application' in prompt_data:
            loan_app = prompt_data['loan_application']
        else:
            loan_app = prompt_data
        
        # Build comprehensive multi-agent prompt
        system_prompt = """You are a professional loan underwriting system consisting of three specialized agents:

1. **Loan Officer**: Reviews application completeness, verifies employment and income
2. **Credit Analyst**: Analyzes credit history, calculates DTI ratio, assesses risk
3. **Risk Manager**: Makes final decision, sets interest rates and terms

Each agent must provide detailed analysis in their area of expertise."""

        user_prompt = f"""Please process this loan application through all three agents:

**Loan Application Data:**
{json.dumps(loan_app, indent=2)}

Generate a comprehensive response in the following JSON format:
{{
    "trajectory_id": "unique-id",
    "agent_outputs": {{
        "loan_officer": {{
            "application_id": "...",
            "applicant_name": "...",
            "employment_status": "...",
            "monthly_income": 0.0,
            "monthly_debts": 0.0,
            "loan_amount": 0.0,
            "loan_purpose": "...",
            "summary": "...",
            "recommendation": "..."
        }},
        "credit_analyst": {{
            "application_id": "...",
            "credit_score": 0,
            "credit_tier": "excellent/good/fair/poor",
            "dti_ratio": 0.0,
            "monthly_income": 0.0,
            "monthly_debts": 0.0,
            "risk_factors": [],
            "risk_score": 0,
            "risk_assessment": "low/medium/high",
            "recommendation": "..."
        }},
        "risk_manager": {{
            "application_id": "...",
            "decision": "APPROVED/DENIED",
            "interest_rate": 0.0,
            "loan_amount": 0.0,
            "term_months": 0,
            "monthly_payment": 0.0,
            "conditions": [],
            "denial_reasons": [],
            "approval_notes": "..."
        }}
    }},
    "decision": "APPROVED/DENIED",
    "workflow_metadata": {{
        "completed_at": "timestamp",
        "duration_ms": 0
    }}
}}

Important Guidelines:
1. Loan Officer must extract and verify all employment and income data
2. Credit Analyst must calculate DTI ratio correctly: (monthly_debts + monthly_payment) / monthly_income
3. Risk Manager must set interest rates based on risk assessment:
   - Low risk: 5-8%
   - Medium risk: 8-12%
   - High risk: 12-18%
4. All agents must reference consistent loan amounts and income figures
5. Decision logic must be consistent across all agents"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def run(self) -> List[Experience]:
        """Run the workflow and calculate rewards using custom reward function"""
        # Format messages for the model
        messages = self.format_messages()
        
        # Generate responses using the model
        experiences = self.model.chat(messages, **self.rollout_args)
        
        # Process each experience
        for exp in experiences:
            try:
                # Calculate reward using our custom reward function
                reward = self.reward_fn(
                    response=exp.response_text,
                    prompt=self.task.raw_task.get('prompt'),
                    truth=self.task.raw_task.get('response'),  # Use response from selected best
                    return_dict=self.is_eval
                )
                
                # Handle dict rewards for eval mode
                if isinstance(reward, dict):
                    exp.metrics = reward
                    # For training, use the single reward value
                    reward = reward.get('reward', sum(reward.values()))
                
                exp.reward = reward
                
                # Add additional metrics
                if not hasattr(exp, 'info') or exp.info is None:
                    exp.info = {}
                exp.info['workflow'] = 'loan_underwriting'
                exp.info['reward_type'] = 'custom_loan_underwriting'
                exp.info['has_truth'] = bool(self.task.raw_task.get('response'))
                exp.info['truth_reward'] = self.task.raw_task.get('reward', 0.0)
                
                logger.debug(f"Calculated reward: {reward} for response length: {len(exp.response_text)}")
                
            except Exception as e:
                logger.error(f"Error calculating reward: {e}")
                exp.reward = 0.0
                if self.is_eval:
                    exp.metrics = {"reward": 0.0}
        
        return experiences