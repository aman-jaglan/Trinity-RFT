"""
Loan Failure Learning Reward Functions

Rule-based reward function that evaluates loan underwriting responses independently
with verifiable calculations and asymmetric rewards for imbalanced dataset.
"""

import json
import re
import random
import logging
from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn


logger = logging.getLogger(__name__)


@REWARD_FUNCTIONS.register_module("loan_underwriting_reward")
class LoanUnderwritingReward(RewardFn):
    """Rule-based reward function with verifiable calculations for loan underwriting"""
    
    def __init__(self):
        super().__init__()
        # Asymmetric reward scaling for imbalanced dataset (79% reward=4)
        # Higher rewards for minority classes to prevent mode collapse
        self.reward_scale = {
            1: 0.0,    # Worst (minority) - clear fail
            2: 0.15,   # Poor (minority) - some violations  
            3: 0.3,    # Fair (minority) - minor issues
            4: 0.4,    # Good (majority) - lower weight to balance
            5: 1.0     # Excellent (minority) - perfect execution
        }
        
    def __call__(self, response, prompt=None, truth=None, return_dict=False):
        """
        Calculate reward based on verifiable loan underwriting criteria.
        
        Each response is evaluated independently based on:
        1. Valid JSON structure and required fields
        2. Correct DTI calculation
        3. Appropriate interest rate based on risk
        4. Consistent decision logic across agents
        5. Proper risk assessment
        """
        # Log that reward function is being called (helpful for debugging)
        logger.debug(f"LoanUnderwritingReward called with response length: {len(str(response))}, "
                    f"truth: {truth}, return_dict: {return_dict}")
        
        try:
            # Parse response
            if isinstance(response, str):
                # Remove any markdown code blocks if present
                response = re.sub(r'```json\s*', '', response)
                response = re.sub(r'```\s*$', '', response)
                response_data = json.loads(response)
            else:
                response_data = response
                
            # Initialize scoring components
            score_components = {
                "structure": 0.0,
                "calculations": 0.0,
                "risk_assessment": 0.0,
                "decision_logic": 0.0,
                "consistency": 0.0
            }
            
            # 1. Check JSON structure (20% weight)
            structure_score = self._check_structure(response_data)
            score_components["structure"] = structure_score * 0.2
            
            # Early exit for invalid structure
            if structure_score < 0.5:
                final_score = sum(score_components.values())
                logger.debug(f"Early exit due to invalid structure: {final_score}")
                if return_dict:
                    return score_components
                return final_score
            
            # Extract agent outputs
            agents = response_data.get("agent_outputs", {})
            loan_officer = agents.get("loan_officer", {})
            credit_analyst = agents.get("credit_analyst", {})
            risk_manager = agents.get("risk_manager", {})
            
            # 2. Verify calculations (25% weight) - CRITICAL for verifiable rewards
            calc_score = self._verify_calculations(loan_officer, credit_analyst, risk_manager)
            score_components["calculations"] = calc_score * 0.25
            
            # 3. Assess risk evaluation (20% weight)
            risk_score = self._assess_risk_evaluation(credit_analyst, loan_officer)
            score_components["risk_assessment"] = risk_score * 0.2
            
            # 4. Check decision logic (20% weight)
            decision_score = self._check_decision_logic(
                credit_analyst, risk_manager, response_data.get("decision", "")
            )
            score_components["decision_logic"] = decision_score * 0.2
            
            # 5. Verify consistency (15% weight)
            consistency_score = self._check_consistency(agents, response_data.get("decision", ""))
            score_components["consistency"] = consistency_score * 0.15
            
            # Calculate final score
            final_score = sum(score_components.values())
            
            # Apply asymmetric reward transformation based on score bands
            if final_score >= 0.9:
                reward_class = 5  # Excellent
            elif final_score >= 0.75:
                reward_class = 4  # Good (majority class - lower reward)
            elif final_score >= 0.6:
                reward_class = 3  # Fair
            elif final_score >= 0.4:
                reward_class = 2  # Poor
            else:
                reward_class = 1  # Worst
                
            # Use asymmetric rewards to handle imbalance
            final_reward = self.reward_scale[reward_class]
            
            # Add noise to prevent identical rewards for similar responses
            noise = random.uniform(-0.02, 0.02)
            final_reward = max(0.0, min(1.0, final_reward + noise))
            
            logger.debug(f"Reward calculation: raw_score={final_score:.3f}, "
                       f"class={reward_class}, final={final_reward:.3f}")
            
            if return_dict:
                # For eval mode, return only the final reward to avoid double counting
                # when workflow sums dict values
                return {"reward": final_reward}
                
            return final_reward
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Error evaluating response: {type(e).__name__}: {e}")
            if return_dict:
                return {"reward": 0.0}  # Consistent with main return
            return 0.0
    
    def _check_structure(self, response_data):
        """Check if response has valid structure with required fields"""
        required_fields = ["agent_outputs", "decision", "trajectory_id"]
        score = 0.0
        
        # Check top-level fields
        for field in required_fields:
            if field in response_data:
                score += 0.2
                
        # Check agent structure
        agents = response_data.get("agent_outputs", {})
        required_agents = ["loan_officer", "credit_analyst", "risk_manager"]
        
        for agent in required_agents:
            if agent in agents and isinstance(agents[agent], dict):
                score += 0.13
                
        # Check for critical fields in each agent
        if "loan_officer" in agents:
            lo = agents["loan_officer"]
            if all(field in lo for field in ["monthly_income", "monthly_debts", "loan_amount"]):
                score += 0.1
                
        if "credit_analyst" in agents:
            ca = agents["credit_analyst"]
            if all(field in ca for field in ["dti_ratio", "credit_score", "risk_assessment"]):
                score += 0.1
                
        if "risk_manager" in agents:
            rm = agents["risk_manager"]
            if all(field in rm for field in ["decision", "interest_rate"]):
                score += 0.1
                
        return min(score, 1.0)
    
    def _verify_calculations(self, loan_officer, credit_analyst, risk_manager):
        """Verify mathematical calculations are correct"""
        score = 0.0
        
        try:
            # Extract values
            monthly_income = float(loan_officer.get("monthly_income", 0))
            monthly_debts = float(loan_officer.get("monthly_debts", 0))
            loan_amount = float(loan_officer.get("loan_amount", 0))
            
            # Verify DTI calculation
            if monthly_income > 0:
                # Calculate expected monthly payment (rough estimate)
                # Check if interest rate and term are provided, don't use defaults
                if "interest_rate" not in risk_manager or "term_months" not in risk_manager:
                    # Can't verify calculations without these fields
                    return score
                    
                interest_rate = float(risk_manager.get("interest_rate", 0)) / 100 / 12
                term_months = int(risk_manager.get("term_months", 0))
                
                if interest_rate > 0 and term_months > 0:
                    # Monthly payment formula
                    monthly_payment = (loan_amount * interest_rate * (1 + interest_rate)**term_months) / \
                                    ((1 + interest_rate)**term_months - 1)
                    
                    # Expected DTI
                    expected_dti = (monthly_debts + monthly_payment) / monthly_income
                    
                    # Check if credit analyst calculated it correctly
                    reported_dti = float(credit_analyst.get("dti_ratio", 0))
                    
                    # Allow 5% tolerance for rounding
                    if abs(reported_dti - expected_dti) < 0.05:
                        score += 0.5  # Correct DTI calculation
                    elif abs(reported_dti - expected_dti) < 0.1:
                        score += 0.3  # Close but not exact
                    else:
                        # Check if they at least tried to calculate DTI
                        if reported_dti > 0:
                            score += 0.1
                            
                # Verify interest rate is reasonable for risk level
                risk_assessment = credit_analyst.get("risk_assessment", "").lower()
                interest_rate_annual = float(risk_manager.get("interest_rate", 0))
                
                if risk_assessment == "low" and 5 <= interest_rate_annual <= 8:
                    score += 0.25
                elif risk_assessment == "medium" and 8 <= interest_rate_annual <= 12:
                    score += 0.25  
                elif risk_assessment == "high" and 12 <= interest_rate_annual <= 18:
                    score += 0.25
                elif interest_rate_annual > 0:  # At least provided a rate
                    score += 0.1
                    
                # Check monthly payment calculation
                if "monthly_payment" in risk_manager and monthly_payment > 0:
                    reported_payment = float(risk_manager.get("monthly_payment", 0))
                    if abs(reported_payment - monthly_payment) / monthly_payment < 0.05:
                        score += 0.25
                        
        except (ValueError, ZeroDivisionError, KeyError) as e:
            # Calculation errors get minimal score
            logger.debug(f"Calculation error in _verify_calculations: {e}")
            
        return min(score, 1.0)
    
    def _assess_risk_evaluation(self, credit_analyst, loan_officer):
        """Assess quality of risk evaluation"""
        score = 0.0
        
        # Check credit score evaluation
        credit_score = credit_analyst.get("credit_score", 0)
        if isinstance(credit_score, (int, float)):
            credit_score = int(credit_score)
            credit_tier = credit_analyst.get("credit_tier", "").lower()
            
            # Verify credit tier matches score
            if (credit_score >= 750 and credit_tier == "excellent") or \
               (700 <= credit_score < 750 and credit_tier == "good") or \
               (650 <= credit_score < 700 and credit_tier == "fair") or \
               (credit_score < 650 and credit_tier == "poor"):
                score += 0.3
                
        # Check risk factors identification
        risk_factors = credit_analyst.get("risk_factors", [])
        if isinstance(risk_factors, list) and len(risk_factors) > 0:
            score += 0.2
            
            # Bonus for identifying specific risks
            dti_ratio = float(credit_analyst.get("dti_ratio", 0))
            if dti_ratio > 0.43 and any("dti" in str(risk).lower() for risk in risk_factors):
                score += 0.1
            if credit_score < 650 and any("credit" in str(risk).lower() for risk in risk_factors):
                score += 0.1
                
        # Check risk score is provided and reasonable
        risk_score_val = credit_analyst.get("risk_score", 0)
        if isinstance(risk_score_val, (int, float)) and 1 <= risk_score_val <= 10:
            score += 0.2
            
        # Employment verification mentioned
        employment_status = loan_officer.get("employment_status", "").lower()
        if employment_status and employment_status != "unknown":
            score += 0.1
            
        return min(score, 1.0)
    
    def _check_decision_logic(self, credit_analyst, risk_manager, final_decision):
        """Check if decision follows logical rules"""
        score = 0.0
        
        try:
            # Extract key metrics
            dti_ratio = float(credit_analyst.get("dti_ratio", 0))
            credit_score = int(credit_analyst.get("credit_score", 0))
            risk_assessment = credit_analyst.get("risk_assessment", "").lower()
            rm_decision = risk_manager.get("decision", "").upper()
            
            # Decision should match between risk manager and final
            if rm_decision == final_decision.upper():
                score += 0.3
                
            # Check decision logic based on metrics
            if dti_ratio > 0:
                # High DTI (>43%) should usually result in denial or high rates
                if dti_ratio > 0.43:
                    if rm_decision == "DENIED":
                        score += 0.3
                    elif rm_decision == "APPROVED":
                        # Should have high interest rate
                        interest_rate = float(risk_manager.get("interest_rate", 0))
                        if interest_rate > 12:
                            score += 0.2
                else:
                    score += 0.1  # Reasonable DTI
                    
            # Credit score logic
            if credit_score < 620 and rm_decision == "DENIED":
                score += 0.2
            elif credit_score >= 720 and rm_decision == "APPROVED":
                score += 0.2
            elif 620 <= credit_score < 720:
                score += 0.1  # Gray area - either could be valid
                
            # Risk assessment alignment
            if risk_assessment == "high" and rm_decision == "DENIED":
                score += 0.2
            elif risk_assessment == "low" and rm_decision == "APPROVED":
                score += 0.2
                
        except (ValueError, KeyError) as e:
            logger.debug(f"Error in _check_decision_logic: {e}")
            
        return min(score, 1.0)
    
    def _check_consistency(self, agents, final_decision):
        """Check consistency across all three agents"""
        score = 0.0
        
        # All agents should reference the same loan amount
        loan_amounts = []
        for agent_name, agent_data in agents.items():
            if "loan_amount" in agent_data:
                loan_amounts.append(float(agent_data.get("loan_amount", 0)))
                
        if len(set(loan_amounts)) == 1 and len(loan_amounts) >= 2:
            score += 0.3
            
        # Income should be consistent
        incomes = []
        for agent_name, agent_data in agents.items():
            if "monthly_income" in agent_data:
                incomes.append(float(agent_data.get("monthly_income", 0)))
                
        if len(set(incomes)) == 1 and len(incomes) >= 2:
            score += 0.3
            
        # Recommendations should align with final decision
        recommendations = []
        for agent_name, agent_data in agents.items():
            if "recommendation" in agent_data:
                rec = agent_data["recommendation"].upper()
                if "APPROV" in rec:
                    recommendations.append("APPROVED")
                elif "DENI" in rec or "DECLIN" in rec:
                    recommendations.append("DENIED")
                    
        if recommendations:
            if all(rec == "APPROVED" for rec in recommendations) and final_decision == "APPROVED":
                score += 0.4
            elif all(rec == "DENIED" for rec in recommendations) and final_decision == "DENIED":
                score += 0.4
            elif len(set(recommendations)) > 1:  # Mixed recommendations
                score += 0.2  # Partial credit for manual review case
                
        return min(score, 1.0)