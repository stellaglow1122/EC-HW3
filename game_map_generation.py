import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

MAP_SIZE = 46
NUM_HUMANS = 5
NUM_GENERATIONS = 1000
POPULATION_SIZE = 100
MUTATION_RATE = 0.1

# Initialize the map first for faster constringency
def initialize_population(size):
    population = []
    for _ in range(size):
        map_grid = np.random.randint(0, 5, (MAP_SIZE, MAP_SIZE))
        
        # Put water in the surrounding area
        map_grid[0, :] = 1
        map_grid[-1, :] = 1
        map_grid[:, 0] = 1
        map_grid[:, -1] = 1

        # Put mountain in the center
        center = (MAP_SIZE // 2, MAP_SIZE // 2)
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                if np.sqrt((y - center[0])**2 + (x - center[1])**2) < MAP_SIZE / 6:
                    map_grid[y, x] = 0

        # Randomly adding rocks and humans
        for _ in range(NUM_HUMANS):
            while True:
                y, x = random.randint(1, MAP_SIZE - 2), random.randint(1, MAP_SIZE - 2)
                if map_grid[y, x] not in [4, 1]:  # Avoiding human on the water
                    map_grid[y, x] = 4
                    break

        population.append(map_grid)
    return population

def fitness_function(map_grid):
    fitness = 0

    # water should be in the surrounding area
    border_water = np.all(map_grid[0, :] == 1) and \
                   np.all(map_grid[-1, :] == 1) and \
                   np.all(map_grid[:, 0] == 1) and \
                   np.all(map_grid[:, -1] == 1)
    fitness += 200 if border_water else -200

    # mountain should be in the center
    center = (MAP_SIZE // 2, MAP_SIZE // 2)
    y, x = np.ogrid[:MAP_SIZE, :MAP_SIZE]
    distance_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    gaussian_mask = np.exp(-distance_from_center**2 / (2 * (MAP_SIZE / 6)**2))
    rock_score = np.sum((map_grid == 0) * gaussian_mask)
    fitness += rock_score * 50

    # grass and rocks
    grass_ratio = np.mean(map_grid == 2)
    rock_ratio = np.mean(map_grid == 3)
    if 0.2 <= grass_ratio <= 0.4:
        fitness += 100
    if 0.05 <= rock_ratio <= 0.15:
        fitness += 100

    # human
    human_positions = np.argwhere(map_grid == 4)
    if len(human_positions) == NUM_HUMANS:
        distances = [np.linalg.norm(human_positions[i] - human_positions[j])
                     for i in range(len(human_positions)) for j in range(i + 1, len(human_positions))]
        fitness += 100 if np.mean(distances) > (MAP_SIZE / 10) else -50
    else:
        fitness -= 200

    return fitness

def genetic_algorithm():
    # Initialize the population
    population = initialize_population(POPULATION_SIZE)

    for generation in range(NUM_GENERATIONS):
        # calculate fitness
        fitness_scores = [fitness_function(individual) for individual in population]

        # selection
        sorted_population = [population[i] for i in np.argsort(fitness_scores)[::-1]]
        next_gen = sorted_population[:POPULATION_SIZE // 2]

        # crossover
        while len(next_gen) < POPULATION_SIZE:
            parent1, parent2 = random.sample(next_gen, 2)
            crossover_point = random.randint(1, MAP_SIZE - 1)
            child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
            next_gen.append(child)

        # mutation
        for individual in next_gen:
            if random.random() < MUTATION_RATE:
                mutate_y = random.randint(0, MAP_SIZE - 1)
                mutate_x = random.randint(0, MAP_SIZE - 1)
                individual[mutate_y, mutate_x] = random.randint(0, 4)

        population = next_gen

        best_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # return the best individual
    best_individual = max(population, key=fitness_function)
    return best_individual

# Run 10 tens to generate 10 images
for i in range(10):
    best_map = genetic_algorithm()
    # File path to save the map
    output_file_path = f"./data/best_map_output_{i}.map"

    # Save the best_map as a plain text file
    np.savetxt(output_file_path, best_map, fmt='%d', delimiter='')

TILE_IMAGES = {
    0: r"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\mountain.png",
    1: r"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\river.png",
    2: r"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\grass.png",
    3: r"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\rock.png", 
    4: r"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\avatar.png"
}
for i in range(10):
    # read number map
    map_file_path = fr"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\data\best_map_output_{i}.map"
    with open(map_file_path, "r") as file:
        raw_data = file.read().strip().splitlines()

    numeric_map = np.array([[int(char) for char in line] for line in raw_data])

    # print("Shape of numeric_map:", numeric_map.shape)

    tile_size = (32, 32) 

    final_image = Image.new('RGB', (numeric_map.shape[1] * tile_size[0], numeric_map.shape[0] * tile_size[1]))

    tile_images = {key: Image.open(value).resize(tile_size) for key, value in TILE_IMAGES.items()}

    for row in range(numeric_map.shape[0]):
        for col in range(numeric_map.shape[1]):
            tile_type = numeric_map[row, col]
            tile_image = tile_images[tile_type]
            final_image.paste(tile_image, (col * tile_size[0], row * tile_size[1]))

    output_image_path = fr"C:\Users\wellm\OneDrive\文件\pythonFiles\ECHW3\output\final_map_image_{i}.png"
    final_image.save(output_image_path)
    #final_image.show() 
