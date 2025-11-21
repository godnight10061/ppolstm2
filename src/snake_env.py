import numpy as np


class SnakeEnv:
    """Simple snake environment with a Gym-like interface."""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self.spawn_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Call reset().")

        if action == 0:  # up
            self.direction = (-1, 0)
        elif action == 1:  # down
            self.direction = (1, 0)
        elif action == 2:  # left
            self.direction = (0, -1)
        elif action == 3:  # right
            self.direction = (0, 1)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self._is_collision(new_head):
            self.done = True
            reward = -1
            return self._get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 1
            self.spawn_food()
        else:
            self.snake.pop()
            reward = 0

        return self._get_state(), reward, self.done, {}

    def spawn_food(self):
        while True:
            food = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if food not in self.snake:
                self.food = food
                break

    def _is_collision(self, position):
        x, y = position
        return (
            x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or position in self.snake
        )

    def _get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            grid[x, y] = 1
        food_x, food_y = self.food
        grid[food_x, food_y] = -1
        return grid.flatten()
