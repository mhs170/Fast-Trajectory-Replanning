import heapq
import numpy as np
import os

# Assuming gridworld_gen.py is in the same directory
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
        self.search()
        return self.expanded_nodes


if __name__ == "__main__":
    expanded_nodes_high_g = []
    expanded_nodes_low_g = []

    for i in range(1, 51):
        filename = f"gridworld_{i}.txt"  
        print(f"\nRunning experiment for {filename}:")
        grid_world = GridWorld.load_from_file(filename)

        repeated_a_star_high_g = RepeatedAStar(grid_world, tie_breaking='high_g')
        high_g_expansions = repeated_a_star_high_g.run_experiment()
        expanded_nodes_high_g.append(high_g_expansions)

        repeated_a_star_low_g = RepeatedAStar(grid_world, tie_breaking='low_g')
        low_g_expansions = repeated_a_star_low_g.run_experiment()
        expanded_nodes_low_g.append(low_g_expansions)

        print(f"Expanded nodes (high g-values) for {filename}: {high_g_expansions}")
        print(f"Expanded nodes (low g-values) for {filename}: {low_g_expansions}")

    print("\nExpanded nodes for all gridworlds:")
    print("Gridworld, High g-values, Low g-values")
    for i in range(50):
        print(f"gridworld_{i+1}.txt, {expanded_nodes_high_g[i]}, {expanded_nodes_low_g[i]}")