import numpy as np
from src import train


class FakeTime:
    def __init__(self):
        self.current = 0.0

    def time(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


def test_train_agent_stops_when_patience_expires(monkeypatch):
    fake_time = FakeTime()

    class StubSnakeEnv:
        def __init__(self, grid_size=10):
            self.state = np.zeros(100, dtype=np.float32)
            self.score = 0

        def reset(self):
            return self.state

        def step(self, action):
            return self.state, 0.0, True, {}

    class StubAgent:
        def __init__(self, *_, **__):
            self.update_calls = 0

        def select_action(self, state, hidden_state=None):
            return 0, hidden_state

        def store_transition(self, reward, done):
            pass

        def update(self):
            self.update_calls += 1
            fake_time.advance(120)

    monkeypatch.setattr(train, "SnakeEnv", StubSnakeEnv)
    monkeypatch.setattr(train, "PPOAgent", StubAgent)

    agent = train.train_agent(patience_seconds=300, time_provider=fake_time.time)

    assert agent.update_calls == 4


def test_train_agent_respects_episode_limit(monkeypatch):
    fake_time = FakeTime()

    class StubSnakeEnv:
        def __init__(self, grid_size=10):
            self.state = np.zeros(100, dtype=np.float32)
            self.score = 0

        def reset(self):
            return self.state

        def step(self, action):
            self.score += 1
            return self.state, 0.0, True, {}

    class StubAgent:
        def __init__(self, *_, **__):
            self.update_calls = 0

        def select_action(self, state, hidden_state=None):
            return 0, hidden_state

        def store_transition(self, reward, done):
            pass

        def update(self):
            self.update_calls += 1
            fake_time.advance(30)

    monkeypatch.setattr(train, "SnakeEnv", StubSnakeEnv)
    monkeypatch.setattr(train, "PPOAgent", StubAgent)

    agent = train.train_agent(
        episodes=3,
        patience_seconds=500,
        time_provider=fake_time.time,
    )

    assert agent.update_calls == 3


def test_train_agent_runs_indefinitely_until_patience(monkeypatch):
    fake_time = FakeTime()
    episode_counter = [0]

    class StubSnakeEnv:
        def __init__(self, grid_size=10):
            self.state = np.zeros(100, dtype=np.float32)
            self.score = 0

        def reset(self):
            episode_counter[0] += 1
            if episode_counter[0] <= 10:
                self.score = episode_counter[0]
            return self.state

        def step(self, action):
            return self.state, 0.0, True, {}

    class StubAgent:
        def __init__(self, *_, **__):
            self.update_calls = 0

        def select_action(self, state, hidden_state=None):
            return 0, hidden_state

        def store_transition(self, reward, done):
            pass

        def update(self):
            self.update_calls += 1
            fake_time.advance(30)

    monkeypatch.setattr(train, "SnakeEnv", StubSnakeEnv)
    monkeypatch.setattr(train, "PPOAgent", StubAgent)

    agent = train.train_agent(patience_seconds=400, time_provider=fake_time.time)

    assert agent.update_calls > 10
