import heapq
import time
import random
from gridworld_gen import GridWorld

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, random.random(), item))  # Ensure random tie-breaking
    
    def get(self):
        return heapq.heappop(self.elements)[2]

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
                return self.reconstruct_path()
            
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
    
    def run_experiment(self):
        start_time = time.time()  
        path = self.search()
        end_time = time.time()  
        runtime = end_time - start_time
        return self.expanded_nodes, runtime, path

def compare_methods():
    results = []
    for i in range(1, 51):
        filename = f"gridworld_{i}.txt"  
        print(f"\nRunning experiment for {filename}:")
        grid_world = GridWorld.load_from_file(filename)

        forward_search = RepeatedAStar(grid_world, forward=True)
        forward_expansions, forward_time, forward_path = forward_search.run_experiment()

        backward_search = RepeatedAStar(grid_world, forward=False)
        backward_expansions, backward_time, backward_path = backward_search.run_experiment()

        results.append((filename, forward_expansions, backward_expansions, forward_time, backward_time))

        print(f"Repeated Forward A* - Expanded Nodes: {forward_expansions}, Runtime: {forward_time:.6f} sec")
        print(f"Repeated Backward A* - Expanded Nodes: {backward_expansions}, Runtime: {backward_time:.6f} sec")

    print("\nFinal Results:")
    print("Gridworld, Forward Expansions, Backward Expansions, Forward Time, Backward Time")
    for result in results:
        print(f"{result[0]}, {result[1]}, {result[2]}, {result[3]:.6f}, {result[4]:.6f}")

if __name__ == "__main__":
    compare_methods()
