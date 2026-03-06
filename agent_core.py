from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentDecision:
    failure_risk: float
    label: int
    action: str
    reason: str

class PredictiveMaintenanceAgent:
    """
    AI Agent = Model prediction + decision policy + action recommendation.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = float(threshold)

    @classmethod
    def from_model_artifact(cls, artifact: Dict[str, Any]) -> "PredictiveMaintenanceAgent":
        threshold = artifact.get("threshold", 0.7)
        return cls(threshold=threshold)

    def decide(self, failure_risk: float) -> AgentDecision:
        label = 1 if failure_risk >= self.threshold else 0

        if label == 1:
            return AgentDecision(
                failure_risk=failure_risk,
                label=1,
                action="CREATE_MAINTENANCE_ALERT",
                reason=f"Risk {failure_risk:.2f} >= threshold {self.threshold:.2f}. Maintenance should be scheduled."
            )

        return AgentDecision(
            failure_risk=failure_risk,
            label=0,
            action="NO_ACTION",
            reason=f"Risk {failure_risk:.2f} < threshold {self.threshold:.2f}. Continue monitoring."
        )
