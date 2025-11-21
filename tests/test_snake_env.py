import pytest
import numpy as np
from src.snake_env import SnakeEnv


class TestSnakeEnv:
    def test_init(self):
        env = SnakeEnv(grid_size=10)
        assert env.grid_size == 10
        assert len(env.snake) == 1
        assert not env.done

    def test_reset(self):
        env = SnakeEnv(grid_size=10)
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (100,)
        assert len(env.snake) == 1
        assert env.score == 0
        assert not env.done

    def test_step_valid_action(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        state, reward, done, info = env.step(0)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_collision_with_wall(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(0, 0)]
        env.direction = (0, 1)
        _, reward, done, _ = env.step(0)
        assert done
        assert reward == -1

    def test_food_consumption(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5)]
        env.food = (4, 5)
        initial_length = len(env.snake)
        _, reward, done, _ = env.step(0)
        assert reward == 1
        assert len(env.snake) == initial_length + 1
        assert env.score == 1

    def test_collision_with_self(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5), (5, 4), (5, 3), (5, 2)]
        env.direction = (0, 1)

        env.step(1)  # Move down
        env.step(2)  # Move left
        _, reward, done, _ = env.step(0)  # Move up collides with (5, 5)
        assert done
        assert reward == -1

    def test_action_directions(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.step(0)
        assert env.direction == (-1, 0)
        env.step(1)
        assert env.direction == (1, 0)
        env.step(2)
        assert env.direction == (0, -1)
        env.step(3)
        assert env.direction == (0, 1)
