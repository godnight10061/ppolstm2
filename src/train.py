import numpy as np

try:
    from src.snake_env import SnakeEnv
    from src.ppo_lstm_agent import PPOAgent
except ModuleNotFoundError:
    from snake_env import SnakeEnv
    from ppo_lstm_agent import PPOAgent


def train_agent(episodes=100):
    env = SnakeEnv(grid_size=10)
    agent = PPOAgent(state_dim=100, action_dim=4)

    for episode in range(episodes):
        state = env.reset()
        hidden_state = None
        total_reward = 0

        while True:
            action, hidden_state = agent.select_action(state, hidden_state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(reward, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update()
                break

        print(f"Episode {episode + 1}, Score: {env.score}, Total Reward: {total_reward}")

    return agent


if __name__ == "__main__":
    train_agent(episodes=100)
