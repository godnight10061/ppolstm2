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
        assert state.shape == (11,)  # Updated for new state representation
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
        assert reward == -10  # Updated penalty

    def test_food_consumption(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5)]
        env.food = (4, 5)
        initial_length = len(env.snake)
        _, reward, done, _ = env.step(0)
        assert reward == 10  # Updated reward
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
        assert reward == -10  # Updated penalty

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

    def test_state_representation_includes_danger_indicators(self):
        env = SnakeEnv(grid_size=10)
        state = env.reset()
        # State should include: 
        # - 3 danger indicators (straight, left, right)
        # - 4 direction indicators (up, down, left, right)
        # - 4 food direction indicators (up, down, left, right)
        assert len(state) == 11

    def test_danger_straight_detected(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(0, 5)]
        env.direction = (-1, 0)  # Moving up, wall ahead
        state = env._get_state()
        assert state[0] == 1  # Danger straight ahead

    def test_danger_left_detected(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 0)]
        env.direction = (-1, 0)  # Moving up, wall to the left
        state = env._get_state()
        assert state[1] == 1  # Danger to the left

    def test_danger_right_detected(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 9)]
        env.direction = (-1, 0)  # Moving up, wall to the right
        state = env._get_state()
        assert state[2] == 1  # Danger to the right

    def test_prevent_reverse_direction(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5), (6, 5)]  # Snake moving up
        env.direction = (-1, 0)  # Moving up
        
        # Try to move down (opposite of up)
        env.step(1)
        # Direction should not change to down
        assert env.direction == (-1, 0)

    def test_distance_based_reward(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5)]
        env.food = (3, 5)
        
        initial_dist = abs(5-3) + abs(5-5)
        
        # Move toward food
        _, reward, _, _ = env.step(0)  # Move up
        assert reward > 0  # Should get positive reward for moving closer
        
    def test_step_penalty_applied(self):
        env = SnakeEnv(grid_size=10)
        env.reset()
        env.snake = [(5, 5)]
        env.food = (3, 3)
        
        # Move away from food
        _, reward, done, _ = env.step(1)  # Move down (away from food)
        if not done:
            assert reward < 0  # Should get negative reward
