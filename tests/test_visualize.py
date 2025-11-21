import pytest
import numpy as np
from unittest.mock import MagicMock
from src import visualize


class TestRenderGame:
    def test_render_game_displays_grid(self, monkeypatch, capsys):
        """Test that render_game correctly displays the game state."""
        mock_time = MagicMock()
        monkeypatch.setattr(visualize, "time", mock_time)

        class MockEnv:
            def __init__(self):
                self.grid_size = 5
                self.snake = [(2, 2), (2, 1)]
                self.food = (4, 4)
                self.score = 3

        env = MockEnv()
        visualize.render_game(env, delay=0.05)

        captured = capsys.readouterr()
        assert "■" in captured.out
        assert "●" in captured.out
        assert "Score: 3" in captured.out
        mock_time.sleep.assert_called_once_with(0.05)


class TestLoadAgent:
    def test_load_agent_success(self, monkeypatch, tmp_path):
        """Test successfully loading an agent from file."""
        model_path = tmp_path / "test_model.pth"
        model_path.touch()

        mock_agent = MagicMock()
        mock_agent_class = MagicMock(return_value=mock_agent)

        monkeypatch.setattr(visualize, "PPOAgent", mock_agent_class)

        result = visualize._load_agent(str(model_path), grid_size=10)

        mock_agent_class.assert_called_once_with(state_dim=11, action_dim=4)
        mock_agent.load_model.assert_called_once_with(str(model_path))
        assert result == mock_agent

    def test_load_agent_file_not_found(self, monkeypatch):
        """Test that loading a non-existent model raises FileNotFoundError."""
        mock_agent = MagicMock()
        mock_agent.load_model.side_effect = FileNotFoundError()
        mock_agent_class = MagicMock(return_value=mock_agent)

        monkeypatch.setattr(visualize, "PPOAgent", mock_agent_class)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            visualize._load_agent("nonexistent.pth", grid_size=10)


class TestVisualizeAgent:
    def test_visualize_agent_runs_single_episode(self, monkeypatch):
        """Test that visualize_agent runs one episode and returns score."""
        step_count = [0]

        class MockEnv:
            def __init__(self, grid_size):
                self.grid_size = grid_size
                self.snake = [(5, 5)]
                self.food = (3, 3)
                self.score = 0
                self.done = False

            def reset(self):
                self.done = False
                self.score = 0
                step_count[0] = 0
                return np.zeros(11)

            def step(self, action):
                step_count[0] += 1
                if step_count[0] >= 3:
                    self.done = True
                    self.score = 5
                return np.zeros(11), 0.0, self.done, {}

        class MockAgent:
            def __init__(self, state_dim, action_dim):
                pass

            def load_model(self, path):
                pass

            def select_action(self, state, hidden_state=None):
                return 1, hidden_state

        def mock_render(env, delay):
            pass

        monkeypatch.setattr(visualize, "SnakeEnv", MockEnv)
        monkeypatch.setattr(visualize, "PPOAgent", MockAgent)
        monkeypatch.setattr(visualize, "render_game", mock_render)

        score = visualize.visualize_agent("model.pth", continuous=False)

        assert score == 5
        assert step_count[0] == 3

    def test_visualize_agent_respects_max_steps(self, monkeypatch):
        """Test that max_steps parameter stops visualization early."""
        step_count = [0]

        class MockEnv:
            def __init__(self, grid_size):
                self.grid_size = grid_size
                self.snake = [(5, 5)]
                self.food = (3, 3)
                self.score = 2
                self.done = False

            def reset(self):
                self.done = False
                step_count[0] = 0
                return np.zeros(11)

            def step(self, action):
                step_count[0] += 1
                self.score = step_count[0]
                return np.zeros(11), 0.0, False, {}

        class MockAgent:
            def __init__(self, state_dim, action_dim):
                pass

            def load_model(self, path):
                pass

            def select_action(self, state, hidden_state=None):
                return 0, hidden_state

        def mock_render(env, delay):
            pass

        monkeypatch.setattr(visualize, "SnakeEnv", MockEnv)
        monkeypatch.setattr(visualize, "PPOAgent", MockAgent)
        monkeypatch.setattr(visualize, "render_game", mock_render)

        score = visualize.visualize_agent("model.pth", max_steps=5)

        assert step_count[0] == 5
        assert score == 5

    def test_visualize_agent_handles_keyboard_interrupt(self, monkeypatch):
        """Test that visualize_agent returns score on KeyboardInterrupt."""

        class MockEnv:
            def __init__(self, grid_size):
                self.grid_size = grid_size
                self.snake = [(5, 5)]
                self.food = (3, 3)
                self.score = 7
                self.done = False

            def reset(self):
                return np.zeros(11)

            def step(self, action):
                raise KeyboardInterrupt()

        class MockAgent:
            def __init__(self, state_dim, action_dim):
                pass

            def load_model(self, path):
                pass

            def select_action(self, state, hidden_state=None):
                return 0, hidden_state

        def mock_render(env, delay):
            pass

        monkeypatch.setattr(visualize, "SnakeEnv", MockEnv)
        monkeypatch.setattr(visualize, "PPOAgent", MockAgent)
        monkeypatch.setattr(visualize, "render_game", mock_render)

        score = visualize.visualize_agent("model.pth")

        assert score == 0

    def test_visualize_agent_uses_default_model_path(self, monkeypatch):
        """Test that visualize_agent uses default model path."""
        load_calls = []

        class MockEnv:
            def __init__(self, grid_size):
                self.grid_size = grid_size
                self.snake = [(5, 5)]
                self.food = (3, 3)
                self.score = 1
                self.done = False

            def reset(self):
                return np.zeros(11)

            def step(self, action):
                self.done = True
                return np.zeros(11), 0.0, True, {}

        class MockAgent:
            def __init__(self, state_dim, action_dim):
                pass

            def load_model(self, path):
                load_calls.append(path)

            def select_action(self, state, hidden_state=None):
                return 0, hidden_state

        def mock_render(env, delay):
            pass

        monkeypatch.setattr(visualize, "SnakeEnv", MockEnv)
        monkeypatch.setattr(visualize, "PPOAgent", MockAgent)
        monkeypatch.setattr(visualize, "render_game", mock_render)

        visualize.visualize_agent()

        assert load_calls == ["best_model.pth"]

    def test_visualize_agent_continuous_mode(self, monkeypatch):
        """Test continuous mode runs multiple episodes."""
        episode_count = [0]

        class MockEnv:
            def __init__(self, grid_size):
                self.grid_size = grid_size
                self.snake = [(5, 5)]
                self.food = (3, 3)
                self.score = 1
                self.done = False

            def reset(self):
                episode_count[0] += 1
                if episode_count[0] > 2:
                    raise KeyboardInterrupt()
                return np.zeros(11)

            def step(self, action):
                self.done = True
                return np.zeros(11), 0.0, True, {}

        class MockAgent:
            def __init__(self, state_dim, action_dim):
                pass

            def load_model(self, path):
                pass

            def select_action(self, state, hidden_state=None):
                return 0, hidden_state

        def mock_render(env, delay):
            pass

        monkeypatch.setattr(visualize, "SnakeEnv", MockEnv)
        monkeypatch.setattr(visualize, "PPOAgent", MockAgent)
        monkeypatch.setattr(visualize, "render_game", mock_render)

        visualize.visualize_agent("model.pth", continuous=True)

        assert episode_count[0] == 3
