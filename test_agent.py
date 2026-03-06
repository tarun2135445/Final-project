import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_core import AgentDecision, PredictiveMaintenanceAgent


class TestAgentDecision:
    def test_fields_accessible(self):
        d = AgentDecision(failure_risk=0.9, label=1, action="CREATE_MAINTENANCE_ALERT", reason="high")
        assert d.failure_risk == 0.9
        assert d.label == 1

    def test_no_action_label(self):
        d = AgentDecision(failure_risk=0.1, label=0, action="NO_ACTION", reason="low")
        assert d.label == 0
        assert d.action == "NO_ACTION"


class TestPredictiveMaintenanceAgent:
    def test_default_threshold(self):
        agent = PredictiveMaintenanceAgent()
        assert agent.threshold == 0.7

    def test_custom_threshold(self):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        assert agent.threshold == 0.5

    def test_from_model_artifact_uses_threshold(self):
        artifact = {"threshold": 0.3, "model": None, "feature_cols": []}
        agent = PredictiveMaintenanceAgent.from_model_artifact(artifact)
        assert agent.threshold == pytest.approx(0.3)

    def test_from_model_artifact_default_threshold(self):
        agent = PredictiveMaintenanceAgent.from_model_artifact({})
        assert agent.threshold == 0.7

    def test_decide_above_threshold_is_failure(self):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        decision = agent.decide(0.8)
        assert decision.label == 1
        assert decision.action == "CREATE_MAINTENANCE_ALERT"

    def test_decide_below_threshold_is_normal(self):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        decision = agent.decide(0.2)
        assert decision.label == 0
        assert decision.action == "NO_ACTION"

    def test_decide_at_threshold_is_failure(self):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        decision = agent.decide(0.5)
        assert decision.label == 1

    def test_decide_returns_agent_decision(self):
        agent = PredictiveMaintenanceAgent()
        decision = agent.decide(0.9)
        assert isinstance(decision, AgentDecision)

    def test_reason_contains_risk_value(self):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        decision = agent.decide(0.75)
        assert "0.75" in decision.reason or "0.7" in decision.reason

    @pytest.mark.parametrize("risk,expected_label", [
        (0.0, 0), (0.49, 0), (0.5, 1), (0.99, 1), (1.0, 1),
    ])
    def test_boundary_risks(self, risk, expected_label):
        agent = PredictiveMaintenanceAgent(threshold=0.5)
        assert agent.decide(risk).label == expected_label
