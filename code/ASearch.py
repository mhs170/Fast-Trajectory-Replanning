import heapq
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from gridworld_gen import GridWorld

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, node, priority):
        heapq.heappush(self.elements, (priority, -node.g_cost, node.x, node.y, node))  # Ensure deterministic tie-breaking
    
    def get(self):
        _, _, _, _, node = heapq.heappop(self.elements)  # Unpack correctly to extract the node
        return node

class RepeatedAStar:
    def __init__(self, gridworld, forward=True):
        self.grid = gridworld.grid
        self.start = gridworld.start_node if forward else gridworld.target_node
        self.goal = gridworld.target_node if forward else gridworld.start_node
        self.forward = forward
        self.open_list = PriorityQueue()
        self.closed_list = set()
        self.initialize_nodes()
        self.expanded_nodes = 0
        self.path = []
    
    def initialize_nodes(self):
        for row in self.grid:
            for node in row:
                node.g_cost = float('inf')
                node.h_cost = self.manhattan_distance(node, self.goal)
                node.parent = None
    
    def manhattan_distance(self, node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)
    
    def get_priority(self, node):
        return node.g_cost + node.h_cost - node.g_cost * 1e-6  # Favor larger g-values
    
    def search(self):
        self.start.g_cost = 0
        self.open_list.put(self.start, self.get_priority(self.start))
        
        while not self.open_list.empty():
            current = self.open_list.get()
            self.expanded_nodes += 1
            
            if current == self.goal:
                self.path = self.reconstruct_path()
                return self.path
            
            if (current.x, current.y) not in self.closed_list:
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
        current = self.goal
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
            if (x, y) != (self.start.x, self.start.y) and (x, y) != (self.goal.x, self.goal.y):
                grid_matrix[x, y] = [0.0, 0.0, 1.0]  # Blue for path
        
        grid_matrix[self.start.x, self.start.y] = [0.5, 1.0, 0.0]  # Lime Green for start node
        grid_matrix[self.goal.x, self.goal.y] = [1.0, 0.0, 0.0]  # Red for goal node
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid_matrix, origin='upper')

        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        
        plt.title("Repeated A* Path Visualization - " + ("Forward" if self.forward else "Reverse"))
        plt.show()
    
    
    def run_experiment(self):
        start_time = time.perf_counter()  # Higher precision timer
        path = self.search()
        end_time = time.perf_counter()  
        runtime = end_time - start_time
        success = path is not None
        return self.expanded_nodes, runtime, success, path

def run_experiment():
    grid_id = int(input("Enter gridworld number (1-50): "))
    filename = f"gridworld_{grid_id}.txt"
    
    grid_world = GridWorld.load_from_file(filename)
    search_type = input("Select search type (1: Forward A*, 2: Reverse A*): ")
    forward = True if search_type == '1' else False
    
    repeated_a_star = RepeatedAStar(grid_world, forward=forward)
    expanded_nodes, runtime, success, path = repeated_a_star.run_experiment()
    
    result_status = "Path Found" if success else "No Path Found"
    print(f"Grid: {filename}")
    print(f"Search Type: {'Forward A*' if forward else 'Reverse A*'}")
    print(f"Expanded Nodes: {expanded_nodes}")
    print(f"Runtime: {runtime:.6f} sec")
    print(f"Result: {result_status}")
    
    if success:
        repeated_a_star.visualize_path()

if __name__ == "__main__":
    run_experiment()


