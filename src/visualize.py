import argparse
import sys
import time

try:
    from src.snake_env import SnakeEnv
    from src.ppo_lstm_agent import PPOAgent
except ModuleNotFoundError:  # pragma: no cover - fallback for local execution
    from snake_env import SnakeEnv
    from ppo_lstm_agent import PPOAgent

DEFAULT_MODEL_PATH = "best_model.pth"


def render_game(env, delay=0.1):
    """Render the current state of the environment in the terminal."""
    print("\033[H\033[J", end="")

    grid = [["·" for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    for index, (x, y) in enumerate(env.snake):
        cell = "■" if index == 0 else "□"
        grid[x][y] = cell

    food_x, food_y = env.food
    grid[food_x][food_y] = "●"

    for row in grid:
        print(" ".join(row))
    print(f"\nScore: {env.score}")
    time.sleep(delay)


def _load_agent(model_path, grid_size):
    agent = PPOAgent(state_dim=11, action_dim=4)
    try:
        agent.load_model(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run training before visualization."
        ) from exc
    return agent


def visualize_agent(
    model_path=DEFAULT_MODEL_PATH,
    render_delay=0.1,
    grid_size=10,
    max_steps=None,
    continuous=False,
):
    """Visualize the best trained agent playing the Snake game."""
    env = SnakeEnv(grid_size=grid_size)
    agent = _load_agent(model_path, grid_size)

    final_score = 0
    try:
        while True:
            state = env.reset()
            hidden_state = None
            render_game(env, delay=render_delay)
            steps = 0

            while True:
                action, hidden_state = agent.select_action(state, hidden_state)
                state, _, done, _ = env.step(action)
                render_game(env, delay=render_delay)
                steps += 1

                if done or (max_steps is not None and steps >= max_steps):
                    final_score = env.score
                    print(f"Episode finished with score: {final_score}")
                    break

            if not continuous:
                break

        return final_score
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
        return final_score


def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO Snake agent")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the trained model file")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between frames in seconds")
    parser.add_argument("--grid-size", type=int, default=10, help="Environment grid size")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to visualize before stopping",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Restart automatically after an episode finishes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        visualize_agent(
            model_path=args.model,
            render_delay=args.delay,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            continuous=args.continuous,
        )
    except FileNotFoundError as exc:  # pragma: no cover - user feedback path
        print(exc)
        sys.exit(1)
