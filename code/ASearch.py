import heapq
import random
import time

class GridWorld:
    def __init__(self, size=101, obstacle_prob=0.3):
        self.size = size
        self.grid = [[0 if random.random() > obstacle_prob else 1 for _ in range(size)] for _ in range(size)]
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.grid[self.start[0]][self.start[1]] = 0  # Ensure start is unblocked
        self.grid[self.goal[0]][self.goal[1]] = 0  # Ensure goal is unblocked
        self.known_grid = [[0 if self.grid[x][y] == 0 else 1 for y in range(size)] for x in range(size)]
    
    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x][y] == 0
    
    def update_knowledge(self, pos):
        x, y = pos
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.known_grid[nx][ny] = self.grid[nx][ny]
    
    def neighbors(self, pos):
        x, y = pos
        return [(nx, ny) for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] if self.is_valid((nx, ny))]
    
    def display(self):
        for row in self.grid:
            print(" ".join("#" if cell == 1 else "." for cell in row))
        print()

class RepeatedAStar:
    def __init__(self, gridworld, forward=True, tie_break='high-g'):
        self.gridworld = gridworld
        self.forward = forward
        self.start, self.goal = (gridworld.start, gridworld.goal) if forward else (gridworld.goal, gridworld.start)
        self.tie_break = tie_break
        self.current_pos = self.start
        self.path = []
    
    def heuristic(self, pos):
        x1, y1 = pos
        x2, y2 = self.goal
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
    
    def a_star_search(self):
        g = {self.current_pos: 0}
        open_set = [(self.heuristic(self.current_pos), self.current_pos)]
        came_from = {}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.goal:
                return self.reconstruct_path(came_from)
            
            for neighbor in [neighbor for neighbor in self.gridworld.neighbors(current) if self.gridworld.known_grid[neighbor[0]][neighbor[1]] == 0]:
                tentative_g = g.get(current, float('inf')) + 1
                if tentative_g < g.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g[neighbor] = tentative_g
                    f_value = tentative_g + self.heuristic(neighbor)
                    priority = (f_value, -tentative_g) if self.tie_break == 'high-g' else (f_value, tentative_g)
                    heapq.heappush(open_set, (priority, neighbor))
        return []  # No path found
    
    def reconstruct_path(self, came_from):
        path = []
        current = self.goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.current_pos)
        path.reverse()
        return path
    
    def search_and_move(self):
        while self.current_pos != self.goal:
            self.gridworld.update_knowledge(self.current_pos)
            self.path = self.a_star_search()
            if not self.path:
                print("No path to the goal found.")
                return
            
            for step in self.path:
                if self.gridworld.grid[step[0]][step[1]] == 1:  # Recalculate if obstacle found
                    break
                self.current_pos = step
                if self.current_pos == self.goal:
                    print("Reached the goal!")
                    return

def compare_methods(size=101, trials=10):
    results = {
        'Repeated Forward A* (High G)': {'time': [], 'expansions': []},
        'Repeated Forward A* (Low G)': {'time': [], 'expansions': []},
        'Repeated Backward A* (High G)': {'time': [], 'expansions': []},
        'Repeated Backward A* (Low G)': {'time': [], 'expansions': []}
    }
    
    for _ in range(trials):
        grid = GridWorld(size)
        
        for tie_break in ['high-g', 'low-g']:
            start_time = time.time()
            forward_search = RepeatedAStar(grid, forward=True, tie_break=tie_break)
            forward_search.search_and_move()
            results[f'Repeated Forward A* ({"High G" if tie_break == "high-g" else "Low G"})']['time'].append(time.time() - start_time)
            
            start_time = time.time()
            backward_search = RepeatedAStar(grid, forward=False, tie_break=tie_break)
            backward_search.search_and_move()
            results[f'Repeated Backward A* ({"High G" if tie_break == "high-g" else "Low G"})']['time'].append(time.time() - start_time)
    
    for key in results:
        avg_time = sum(results[key]['time']) / trials
        print(f"{key}: Avg Time = {avg_time:.5f}s")
    
# Run the comparison
compare_methods()
