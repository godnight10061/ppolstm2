import numpy as np


class SnakeEnv:
    """Snake environment with improved state representation and reward shaping."""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self.spawn_food()
        self.done = False
        self.score = 0
        self.prev_distance = self._get_food_distance()
        return self._get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Call reset().")

        new_direction = self._action_to_direction(action)
        
        if not self._is_opposite_direction(new_direction):
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self._is_collision(new_head):
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
            self.prev_distance = self._get_food_distance()
        else:
            self.snake.pop()
            current_distance = self._get_food_distance()
            if current_distance < self.prev_distance:
                reward = 1
            else:
                reward = -1
            self.prev_distance = current_distance

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

    def _action_to_direction(self, action):
        if action == 0:  # up
            return (-1, 0)
        elif action == 1:  # down
            return (1, 0)
        elif action == 2:  # left
            return (0, -1)
        elif action == 3:  # right
            return (0, 1)
        return self.direction

    def _is_opposite_direction(self, new_direction):
        if len(self.snake) < 2:
            return False
        return (
            self.direction[0] == -new_direction[0] and 
            self.direction[1] == -new_direction[1]
        )

    def _get_food_distance(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def _get_danger_in_direction(self, direction):
        head = self.snake[0]
        next_pos = (head[0] + direction[0], head[1] + direction[1])
        return 1 if self._is_collision(next_pos) else 0

    def _get_relative_direction(self, direction):
        dx, dy = self.direction
        if dx == -1:  # moving up
            if direction == 'left':
                return (0, -1)
            elif direction == 'right':
                return (0, 1)
        elif dx == 1:  # moving down
            if direction == 'left':
                return (0, 1)
            elif direction == 'right':
                return (0, -1)
        elif dy == -1:  # moving left
            if direction == 'left':
                return (1, 0)
            elif direction == 'right':
                return (-1, 0)
        elif dy == 1:  # moving right
            if direction == 'left':
                return (-1, 0)
            elif direction == 'right':
                return (1, 0)
        return (0, 0)

    def _get_state(self):
        head = self.snake[0]
        
        danger_straight = self._get_danger_in_direction(self.direction)
        danger_left = self._get_danger_in_direction(self._get_relative_direction('left'))
        danger_right = self._get_danger_in_direction(self._get_relative_direction('right'))
        
        dir_up = 1 if self.direction == (-1, 0) else 0
        dir_down = 1 if self.direction == (1, 0) else 0
        dir_left = 1 if self.direction == (0, -1) else 0
        dir_right = 1 if self.direction == (0, 1) else 0
        
        food_up = 1 if self.food[0] < head[0] else 0
        food_down = 1 if self.food[0] > head[0] else 0
        food_left = 1 if self.food[1] < head[1] else 0
        food_right = 1 if self.food[1] > head[1] else 0
        
        state = np.array([
            danger_straight,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            food_up,
            food_down,
            food_left,
            food_right,
        ], dtype=np.float32)
        
        return state
