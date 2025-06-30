"""
Loan Failure Learning Workflows

Custom workflows for training loan agents from failure trajectories.
"""

from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow


@WORKFLOWS.register_module("loan_underwriting_workflow")
class LoanUnderwritingWorkflow(SimpleWorkflow):
    """Workflow for learning from loan agent failures"""
    
    def format_messages(self):
        """Override to create failure-aware prompts"""
        
        # Get failure data from task
        task_data = self.task.raw_task
        user_input = task_data.get('user_input', '')
        failed_response = task_data.get('failed_response', '')
        failure_mode = task_data.get('failure_mode', 'unknown')
        
        # Build improvement prompt
        prompt = f"""You are a professional loan advisor. A previous response failed due to: {failure_mode}

User Query: {user_input}
Failed Response: {failed_response}

Generate an improved response that:
1. Avoids the failure mode: {failure_mode}
2. Provides helpful, accurate information
3. Uses professional, inclusive language

Improved Response:"""
        
        # Use Trinity-RFT's message format
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return messages
