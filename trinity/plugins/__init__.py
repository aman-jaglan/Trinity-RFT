"""Trinity plugins for loan underwriting task"""

# Import custom modules to ensure they are registered
from . import loan_underwriting_workflow
from . import loan_rewards
from . import loan_data_reader

__all__ = ['loan_underwriting_workflow', 'loan_rewards', 'loan_data_reader']