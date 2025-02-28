# core algorithm implementation for Repeated A Star and Adaptive A star

import heapq
import numpy as np
import os
import time
import matplotlib.pyplot as plt

try:
    from gridworld_gen import GridWorld
except ImportError:
    print("Error: gridworld_gen.py not found. Make sure it's in the same directory.")
    exit()

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
        self.expanded_nodes = 0
    
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
            return node.g_cost + node.h_cost - node.g_cost * 1e-6  
        else:  # low g-values
            return node.g_cost + node.h_cost + node.g_cost * 1e-6  
    
    def search(self):
        self.start.g_cost = 0
        self.open_list.put(self.start, self.get_priority(self.start))
        
        while not self.open_list.empty():
            current = self.open_list.get()
            if current == self.target:
                return self.reconstruct_path()
            
            if (current.x, current.y) not in self.closed_list:
                self.expanded_nodes += 1
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
        start_time = time.time()  
        self.search()
        end_time = time.time()  
        runtime = end_time - start_time
        return self.expanded_nodes, runtime

class AdaptiveAStar:
    def __init__(self, gridworld, tie_breaking='high_g'):
        self.grid = gridworld.grid
        self.start = gridworld.start_node
        self.target = gridworld.target_node
        self.open_list = PriorityQueue()
        self.closed_list = set()
        self.tie_breaking = tie_breaking
        self.initialize_nodes()
        self.expanded_nodes = 0
        self.path = []
    
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
            return node.g_cost + node.h_cost - node.g_cost * 1e-6  
        else:
            return node.g_cost + node.h_cost + node.g_cost * 1e-6  
    
    def search(self):
        self.start.g_cost = 0
        self.open_list.put(self.start, self.get_priority(self.start))
        
        while not self.open_list.empty():
            current = self.open_list.get()
            if current == self.target:
                self.path = self.reconstruct_path()
                return self.path
            
            if (current.x, current.y) not in self.closed_list:
                self.expanded_nodes += 1
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
    
    def visualize_path(self):
        grid_size = len(self.grid)
        grid_matrix = np.ones((grid_size, grid_size, 3))  # Default white (unblocked cells)
        
        for row in self.grid:
            for node in row:
                if node.node_type == 1:
                    grid_matrix[node.x, node.y] = [0, 0, 0]  # Black for obstacles
        
        for x, y in self.path:
            if (x, y) != (self.start.x, self.start.y) and (x, y) != (self.target.x, self.target.y):
                grid_matrix[x, y] = [0.0, 0.0, 1.0]  # Blue for path
        
        grid_matrix[self.start.x, self.start.y] = [0.5, 1.0, 0.0]  # Lime Green for start node
        grid_matrix[self.target.x, self.target.y] = [1.0, 0.0, 0.0]  # Red for goal node
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_matrix, origin='upper')
        plt.title("A* Path Visualization")
        plt.show()
    
    def run_experiment(self):
        start_time = time.time()  
        path = self.search()
        end_time = time.time()  
        runtime = end_time - start_time
        return self.expanded_nodes, runtime, path



if __name__ == "__main__":
    results = []
    adaptive_faster = 0
    repeated_faster = 0

    for i in range(1, 51):
        filename = f"gridworld_{i}.txt"  
        print(f"\nRunning experiment for {filename}:")
        grid_world = GridWorld.load_from_file(filename)

        repeated_high_g = RepeatedAStar(grid_world, tie_breaking='high_g')
        high_g_expansions, high_g_time = repeated_high_g.run_experiment()

        repeated_low_g = RepeatedAStar(grid_world, tie_breaking='low_g')
        low_g_expansions, low_g_time = repeated_low_g.run_experiment()

        adaptive_a_star = AdaptiveAStar(grid_world, tie_breaking='high_g')
        adaptive_expansions, adaptive_time, path = adaptive_a_star.run_experiment()

        results.append((filename, high_g_expansions, low_g_expansions, adaptive_expansions, high_g_time, low_g_time, adaptive_time))

        print(f"Repeated A* (High G) - Expanded Nodes: {high_g_expansions}, Runtime: {high_g_time:.6f} sec")
        print(f"Repeated A* (Low G) - Expanded Nodes: {low_g_expansions}, Runtime: {low_g_time:.6f} sec")
        print(f"Adaptive A* - Expanded Nodes: {adaptive_expansions}, Runtime: {adaptive_time:.6f} sec")

        if adaptive_time < high_g_time:
            adaptive_faster += 1
        else:
            repeated_faster += 1

    print("\nFinal Results:")
    print("Gridworld, Repeated A* (High G) Expansions, Repeated A* (Low G) Expansions, Adaptive A* Expansions, High G Runtime, Low G Runtime, Adaptive A* Runtime")
    for result in results:
        print(f"{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]:.6f}, {result[5]:.6f}, {result[6]:.6f}")

    print(f"\nAdaptive A* had a faster runtime in {adaptive_faster} out of 50 cases.")
    print(f"Repeated Forward A* (high g-values) had a faster runtime in {repeated_faster} out of 50 cases.")
