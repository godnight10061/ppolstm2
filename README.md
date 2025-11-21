# PPO LSTM Snake Game Agent

A clean and concise implementation of a Proximal Policy Optimization (PPO) agent with LSTM for playing the Snake game, built using Test-Driven Development (TDD).

## Features

- Snake game environment with Gym-like interface
- PPO algorithm with LSTM network for sequential decision making
- Comprehensive test suite
- Clean and modular code structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the agent:
```bash
python src/train.py
```

Run tests:
```bash
pytest tests/
```

## Project Structure

```
src/
├── snake_env.py       # Snake game environment
├── ppo_lstm_agent.py  # PPO LSTM agent
└── train.py           # Training script

tests/
├── test_snake_env.py       # Environment tests
└── test_ppo_lstm_agent.py  # Agent tests
```
