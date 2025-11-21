import pytest
import numpy as np
import torch
from src.ppo_lstm_agent import PPOAgent, PPOPolicyNetwork, PPOMemory


class TestPPOPolicyNetwork:
    def test_init(self):
        network = PPOPolicyNetwork(input_dim=11, action_dim=4, hidden_dim=128)
        assert network.fc1.in_features == 11
        assert network.fc1.out_features == 128
        assert network.lstm.input_size == 128
        assert network.lstm.hidden_size == 128
        assert network.actor.out_features == 4

    def test_forward(self):
        network = PPOPolicyNetwork(input_dim=11, action_dim=4, hidden_dim=128)
        x = torch.randn(1, 5, 11)
        logits, value, hidden_state = network(x)
        assert logits.shape == (1, 5, 4)
        assert value.shape == (1, 5, 1)
        assert hidden_state is not None


class TestPPOMemory:
    def test_init(self):
        memory = PPOMemory()
        assert len(memory.states) == 0
        assert len(memory.actions) == 0
        assert len(memory.rewards) == 0

    def test_clear(self):
        memory = PPOMemory()
        memory.states.append(1)
        memory.actions.append(2)
        memory.rewards.append(3)
        memory.clear()
        assert len(memory.states) == 0
        assert len(memory.actions) == 0
        assert len(memory.rewards) == 0


class TestPPOAgent:
    def test_init(self):
        agent = PPOAgent(state_dim=11, action_dim=4)
        assert isinstance(agent.policy, PPOPolicyNetwork)
        assert isinstance(agent.memory, PPOMemory)
        assert agent.gamma == 0.99

    def test_select_action(self):
        agent = PPOAgent(state_dim=11, action_dim=4)
        state = np.random.randn(11).astype(np.float32)
        action, hidden_state = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 4
        assert hidden_state is not None
        assert len(agent.memory.states) == 1

    def test_store_transition(self):
        agent = PPOAgent(state_dim=11, action_dim=4)
        agent.store_transition(reward=1.0, done=False)
        assert len(agent.memory.rewards) == 1
        assert len(agent.memory.dones) == 1
        assert agent.memory.rewards[0] == 1.0
        assert agent.memory.dones[0] is False

    def test_update(self):
        agent = PPOAgent(state_dim=11, action_dim=4)
        state = np.random.randn(11).astype(np.float32)
        agent.select_action(state)
        agent.store_transition(reward=1.0, done=False)
        agent.select_action(state)
        agent.store_transition(reward=1.0, done=True)
        agent.update()
        assert len(agent.memory.states) == 0
        assert len(agent.memory.rewards) == 0

    def test_higher_entropy_coefficient(self):
        agent = PPOAgent(state_dim=11, action_dim=4)
        # Entropy coefficient should be higher for exploration
        assert hasattr(agent, 'entropy_coef')
        assert agent.entropy_coef >= 0.05  # Should be at least 0.05
