import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output, hidden_state = self.lstm(x, hidden_state)
        logits = self.actor(output)
        value = self.critic(output)
        return logits, value, hidden_state


class PPOMemory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.05,
        device=None,
    ):
        if device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
            if resolved_device.type == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU.",
                    RuntimeWarning,
                )
                resolved_device = torch.device("cpu")
        self.device = resolved_device

        self.policy = PPOPolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

    def select_action(self, state, hidden_state=None):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        logits, value, hidden_state = self.policy(state_tensor, hidden_state)
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        self.memory.states.append(state_tensor)
        self.memory.actions.append(action.item())
        self.memory.log_probs.append(action_log_prob.detach())
        self.memory.values.append(value.detach())

        return action.item(), hidden_state

    def update(self):
        returns = []
        discounted_return = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(self.memory.log_probs)
        values = torch.cat(self.memory.values, dim=1).squeeze(0).squeeze(-1)
        states = torch.cat(self.memory.states, dim=1)
        actions = torch.tensor(self.memory.actions, device=self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            logits, values_pred, _ = self.policy(states)
            logits = logits.squeeze(0)
            values_pred = values_pred.squeeze(0).squeeze(-1)

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs.detach())

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = nn.MSELoss()(values_pred, returns)

            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.memory.clear()

    def store_transition(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)

    def save_model(self, path):
        """Persist the current policy weights to disk."""
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path, map_location=None):
        """Load policy weights from disk for evaluation."""
        if map_location is None:
            map_location = self.device
        state_dict = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
