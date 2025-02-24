import sys
import subprocess
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully!")

# Install required packages if not already installed
install_package("numpy")
install_package("matplotlib")
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Constants
SIZE = 101  
NUM_GRIDS = 50  
BLOCK_CLUSTER_PROB = 0.3  # Probability of starting a block cluster
CLUSTER_GROWTH_PROB = 0.5  # Probability of growing the cluster
SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), "gridworlds")

class GridNode:
    def __init__(self, x, y, node_type = -1):
        self.x = x
        self.y = y
        self.node_type = node_type
        self.g_cost = float('inf')
        self.h_cost = float('inf')
        self.parent = None

    def f_cost(self):
        #Returns the total cost (g + h) for pathfinding.
        return self.g_cost + self.h_cost

class GridWorld:
    def __init__(self):
        self.grid = [[GridNode(x, y) for y in range(SIZE)] for x in range(SIZE)]
        self.start_node = None
        self.target_node = None
        self.generate_maze()
        self.place_start_and_target()
        
    def generate_maze(self):
        stack = []
        start_x, start_y = random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)
        self.grid[start_x][start_y].node_type = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack.pop()
            neighbors = self.get_unvisited_neighbors(x, y)
            if neighbors:
                stack.append((x, y))
                x_new, y_new = random.choice(neighbors)

                if random.random() < BLOCK_CLUSTER_PROB:
                    self.grid[x_new][y_new].node_type = 1
                else:
                    self.grid[x_new][y_new].node_type = 0
                    stack.append((x_new, y_new))

        self.fill_remaining_cells()
    
    def get_unvisited_neighbors(self, x, y):
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] #Right, Left, Down, Up
        random.shuffle(directions)
        for dx, dy in directions:
            x_new, y_new = x + dx, y + dy
            if 0 <= x_new < SIZE and 0 <= y_new < SIZE and self.grid[x_new][y_new].node_type == -1:
                neighbors.append((x_new, y_new))
        return neighbors
    
    def fill_remaining_cells(self):
        for x in range(SIZE):
            for y in range(SIZE):
                if self.grid[x][y].node_type == -1:  
                    self.grid[x][y].node_type = 0  
        
    def place_start_and_target(self):
        empty_cells = [(x, y) for x in range(SIZE) for y in range(SIZE) if self.grid[x][y].node_type == 0]

        if len(empty_cells) < 2:
            raise ValueError("Not enough open spaces to place start and target!")

        start_pos = random.choice(empty_cells)
        empty_cells.remove(start_pos)
        target_pos = random.choice(empty_cells)

        self.start_node = self.grid[start_pos[0]][start_pos[1]]
        self.target_node = self.grid[target_pos[0]][target_pos[1]]
        self.start_node.node_type = 2  # Start node (Lime)
        self.target_node.node_type = 3  # Target node (Red)
    
    def save_to_file(self, filename):
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        filepath = os.path.join(SAVE_DIRECTORY, filename)

        np_grid = np.array([[node.node_type for node in row] for row in self.grid])
        np.savetxt(filepath, np_grid, fmt='%d', delimiter=' ')
        print(f"Grid saved to {filepath}")
    
    def load_from_file(filename):
        filepath = os.path.join(SAVE_DIRECTORY, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: File '{filepath}' not found!")

        np_grid = np.loadtxt(filepath, dtype=int)
        obj = GridWorld()
        for x in range(SIZE):
            for y in range(SIZE):
                obj.grid[x][y].node_type = np_grid[x][y]
                if np_grid[x][y] == 2:
                    obj.start_node = obj.grid[x][y]
                elif np_grid[x][y] == 3:
                    obj.target_node = obj.grid[x][y]

        print(f"Grid loaded from {filepath}")
        return obj

    def visualize(self):
        #Displays the grid with start and target nodes.
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = {0: "white", 1: "black", 2: "lime", 3: "red"}  # Add colors for start & target

        color_grid = np.zeros((SIZE, SIZE, 3))
        for x in range(SIZE):
            for y in range(SIZE):
                color_grid[x, y] = mcolors.to_rgb(cmap[self.grid[x][y].node_type])

        ax.imshow(color_grid, origin='upper')
        ax.set_xticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.title("GridWorld Visualization")
        plt.show()

if __name__ == "__main__":
    action = input("Select an option:\n1: Generate Grids\n2: Display Grid\n")

    if action == "1":
        for i in range(1, NUM_GRIDS + 1):
            grid_world = GridWorld()
            grid_world.save_to_file(f"gridworld_{i}.txt")
        print("All grids generated!")


    existing_grids = sorted([f for f in os.listdir(SAVE_DIRECTORY) if f.startswith("gridworld_")])
    if not existing_grids:
        print("No grid files found! Generate grids first.")
        exit()

    while True:
        try:
            grid_choice = int(input(f"Choose a grid to display (1-{NUM_GRIDS}): "))
            if 1 <= grid_choice <= NUM_GRIDS:
                break
            else:
                print(f"Invalid choice! Enter a number between 1 and {NUM_GRIDS}.")
        except ValueError:
            print("Invalid input! Please enter a number.")

    grid_to_display = GridWorld.load_from_file(f"gridworld_{grid_choice}.txt")
    grid_to_display.visualize()

    # else:
    #     print("Invalid option! Please enter 1 or 2.")
