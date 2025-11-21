import time

try:
    from src.snake_env import SnakeEnv
    from src.ppo_lstm_agent import PPOAgent
except ModuleNotFoundError:
    from snake_env import SnakeEnv
    from ppo_lstm_agent import PPOAgent


def train_agent(episodes=None, patience_seconds=300, time_provider=None):
    env = SnakeEnv(grid_size=10)
    agent = PPOAgent(state_dim=100, action_dim=4)

    if time_provider is None:
        time_provider = time.time

    best_score = float("-inf")
    last_improvement_time = time_provider()
    episode = 0

    while episodes is None or episode < episodes:
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

        episode += 1
        print(f"Episode {episode}, Score: {env.score}, Total Reward: {total_reward}")

        current_time = time_provider()
        if env.score > best_score:
            best_score = env.score
            last_improvement_time = current_time

        if current_time - last_improvement_time >= patience_seconds:
            print(f"Training stopped: no score improvement for {patience_seconds}s")
            break

    return agent


if __name__ == "__main__":
    train_agent()
