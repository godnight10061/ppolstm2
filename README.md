# PPO LSTM Snake Game Agent

A clean and concise implementation of a Proximal Policy Optimization (PPO) agent with LSTM for playing the Snake game, built using Test-Driven Development (TDD).

## Features

- Snake game environment with improved state representation
- PPO algorithm with enhanced LSTM network for sequential decision making
- Reward shaping for faster learning
- Invalid action prevention (no 180-degree turns)
- **GPU acceleration support** - automatically uses CUDA when available
- Comprehensive test suite (35 tests)
- Clean and modular code structure

## Key Improvements

### State Representation (11 features)
- **Danger indicators**: straight, left, right (3 features)
- **Current direction**: up, down, left, right (4 features)
- **Food direction**: up, down, left, right (4 features)

### Reward Shaping
- Eating food: +10
- Collision: -10
- Moving toward food: +1
- Moving away from food: -1

### Network Architecture
- Pre-processing layers before LSTM
- Gradient clipping for stability
- Higher entropy coefficient for exploration (0.05)
- Advantage normalization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the agent:
```bash
python src/train.py
```

Visualize the best trained agent:
```bash
python src/visualize.py
```

Options for visualization:
```bash
python src/visualize.py --model best_model.pth --delay 0.1 --continuous
```

Run tests:
```bash
pytest tests/
```

## Project Structure

```
src/
├── snake_env.py       # Snake game environment with improved state/rewards
├── ppo_lstm_agent.py  # PPO LSTM agent with enhanced architecture
├── train.py           # Training script
└── visualize.py       # Visualization script

tests/
├── test_snake_env.py       # Environment tests (14 tests)
├── test_ppo_lstm_agent.py  # Agent tests (9 tests)
├── test_train.py           # Training tests (4 tests)
└── test_visualize.py       # Visualization tests (8 tests)
```

## Performance

The improved approach provides:
- Dense feedback every step (reward shaping)
- Faster learning convergence
- Better state representation
- More stable training

Expected learning timeline:
- Score 5+: ~500-1000 episodes
- Score 10+: ~2000-5000 episodes
- Continues improving with more training
