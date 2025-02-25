import heapq
import numpy as np
import os
from gridworld_gen import GridWorld

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

class RepeatedAStar:
    def __init__(self, gridworld, tie_breaking='high_g'):
        self.grid = gridworld.grid
        self.start = gridworld.start_node
        self.target = gridworld.target_node
        self.open_list = PriorityQueue()
        self.closed_list = set()
        self.tie_breaking = tie_breaking
        self.initialize_nodes()
    
    def initialize_nodes(self):
        for row in self.grid:
            for node in row:
                node.g_cost = float('inf')
                node.h_cost = self.manhattan_distance(node, self.target)
                node.parent = None
    
    def manhattan_distance(self, node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)
    
    def get_priority(self, node):
        if self.tie_breaking == 'high_g':
            return node.g_cost + node.h_cost - node.g_cost * 1e-6  # Favor higher g-cost
        else:
            return node.g_cost + node.h_cost + node.g_cost * 1e-6  # Favor lower g-cost
    
    def search(self):
        self.start.g_cost = 0
        self.open_list.put(self.start, self.get_priority(self.start))
        
        while not self.open_list.empty():
            current = self.open_list.get()
            if current == self.target:
                return self.reconstruct_path()
            
            self.closed_list.add((current.x, current.y))
            for neighbor in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) in self.closed_list or neighbor.node_type == 1:
                    continue
                tentative_g_cost = current.g_cost + 1
                if tentative_g_cost < neighbor.g_cost:
                    neighbor.g_cost = tentative_g_cost
                    neighbor.parent = current
                    self.open_list.put(neighbor, self.get_priority(neighbor))
        return None
    
    def get_neighbors(self, node):
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                neighbors.append(self.grid[nx][ny])
        return neighbors
    
    def reconstruct_path(self):
        path = []
        current = self.target
        while current:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path
    
    def run_experiment(self):
        print(f"Running Repeated A* with tie-breaking {'high g-values' if self.tie_breaking == 'high_g' else 'low g-values'}")
        path = self.search()
        if path:
            print(f"Path found with {len(path)} steps")
        else:
            print("No path found")
        return len(path) if path else float('inf')

if __name__ == "__main__":
    filename = "gridworld_1.txt"  # Example grid file
    grid_world = GridWorld.load_from_file(filename)
    
    print("Comparing tie-breaking strategies:")
    high_g_expansion = RepeatedAStar(grid_world, tie_breaking='high_g').run_experiment()
    low_g_expansion = RepeatedAStar(grid_world, tie_breaking='low_g').run_experiment()
    
    print(f"Expanded nodes (high g-values): {high_g_expansion}")
    print(f"Expanded nodes (low g-values): {low_g_expansion}")

if __name__ == "__main__":
    for i in range(1, 51):
        filename = f"gridworld_{i}.txt"  
        print(f"\nRunning experiment for {filename}:")
        grid_world = GridWorld.load_from_file(filename)
        high_g_expansion = RepeatedAStar(grid_world, tie_breaking='high_g').run_experiment()
        low_g_expansion = RepeatedAStar(grid_world, tie_breaking='low_g').run_experiment()
        
        print(f"Expanded nodes (high g-values) for {filename}: {high_g_expansion}")
        print(f"Expanded nodes (low g-values) for {filename}: {low_g_expansion}")