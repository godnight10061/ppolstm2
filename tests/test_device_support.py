import warnings

import numpy as np
import pytest
import torch

from src.ppo_lstm_agent import PPOAgent


class TestDeviceSupport:
    def test_agent_exposes_device_and_defaults_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        agent = PPOAgent(state_dim=11, action_dim=4)

        assert hasattr(agent, "device")
        assert agent.device.type == "cpu"

    def test_agent_respects_explicit_device_argument(self):
        agent = PPOAgent(state_dim=11, action_dim=4, device="cpu")
        assert agent.device.type == "cpu"

    def test_agent_falls_back_to_cpu_when_cuda_requested_but_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = PPOAgent(state_dim=11, action_dim=4, device="cuda")

        assert agent.device.type == "cpu"
        assert any("CUDA requested" in str(w.message) for w in caught)

    def test_state_and_logprob_tensors_follow_agent_device(self):
        agent = PPOAgent(state_dim=11, action_dim=4, device="cpu")
        state = np.random.randn(11).astype(np.float32)

        agent.select_action(state)

        assert agent.memory.states[0].device.type == agent.device.type
        assert agent.memory.log_probs[0].device.type == agent.device.type
        assert agent.memory.values[0].device.type == agent.device.type

    def test_tensors_created_during_update_run_on_agent_device(self):
        agent = PPOAgent(state_dim=11, action_dim=4, device="cpu")
        state = np.random.randn(11).astype(np.float32)

        for _ in range(3):
            agent.select_action(state)
            agent.store_transition(reward=1.0, done=False)
        agent.select_action(state)
        agent.store_transition(reward=1.0, done=True)

        agent.update()  # Should not raise device mismatch errors

    def test_load_model_defaults_to_agent_device(self, monkeypatch):
        agent = PPOAgent(state_dim=11, action_dim=4, device="cpu")
        captured = {}

        def fake_load(path, map_location=None):
            captured["map_location"] = map_location
            return agent.policy.state_dict()

        monkeypatch.setattr(torch, "load", fake_load)

        agent.load_model("some_path.pth")

        assert captured["map_location"] == agent.device
