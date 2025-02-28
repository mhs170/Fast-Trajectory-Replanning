# plot grids for parts 2 and 5

import heapq
import numpy as np
import os
import time
from Astar import AdaptiveAStar
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
    grid_number = input("Enter gridworld number (1-50): ")
    filename = f"gridworld_{grid_number}.txt"
    
    # Construct the full path to the gridworld file inside the "gridworlds" folder
    file_path = os.path.join(os.path.dirname(__file__), "gridworlds", filename)

    if not os.path.exists(file_path):
        print(f"Error: Specified gridworld file not found at {file_path}")
        exit()
    
    grid_world = GridWorld.load_from_file(file_path)
    
    print("Choose search algorithm:")
    print("1: Repeated A* (High G)")
    print("2: Repeated A* (Low G)")
    print("3: Adaptive A*")
    choice = input("Enter choice (1/2/3): ")
    
    if choice == "1":
        algorithm = RepeatedAStar(grid_world, tie_breaking='high_g')
    elif choice == "2":
        algorithm = RepeatedAStar(grid_world, tie_breaking='low_g')
    elif choice == "3":
        algorithm = AdaptiveAStar(grid_world, tie_breaking='high_g')
    else:
        print("Invalid choice. Exiting.")
        exit()
    
    expanded_nodes, runtime, path = algorithm.run_experiment()
    
    print(f"Algorithm expanded {expanded_nodes} nodes.")
    print(f"Runtime: {runtime:.6f} seconds.")
    print(f"Path found: {path}")
    

    algorithm.visualize_path()