import random
import math
import numpy as np


# ANGGOTA KELOMPOK:
#1. Gumilar Hari Subagja(103022300137)
#2. Muhammad Ilham Firdaus(103022300138)

# === Parameter GA ===
POP_SIZE = 50
CHROMOSOME_LENGTH = 32   # 16 bit per variabel
GENE_LENGTH = 16
MAX_GENERATIONS = 100
PC = 0.8                 # Probabilitas crossover
PM = 0.02                # Probabilitas mutasi
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

# === Fungsi Objektif (yang akan diminimalkan) ===
def objective_function(x1, x2):
    try:
        term1 = math.sin(x1) * math.cos(x2) * math.tan(x1 + x2)
    except:
        term1 = 0  # Untuk menangani tan tak hingga
    term2 = (3 / 4) * math.exp(1 - math.sqrt(x1 ** 2 + x2 ** 2))
    return -(term1 + term2)

# === Konversi Binary <-> Float ===
def binary_to_float(binary_str, min_val, max_val):
    decimal = int(binary_str, 2)
    return min_val + decimal * (max_val - min_val) / (2 ** len(binary_str) - 1)

def float_to_binary(value, min_val, max_val, length):
    decimal = round((value - min_val) * (2 ** length - 1) / (max_val - min_val))
    return format(decimal, f'0{length}b')

# === Inisialisasi Populasi ===
def initialize_population():
    return [''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH)) for _ in range(POP_SIZE)]

# === Dekode kromosom jadi x1, x2 ===
def decode_chromosome(chromosome):
    x1_bin = chromosome[:GENE_LENGTH]
    x2_bin = chromosome[GENE_LENGTH:]
    x1 = binary_to_float(x1_bin, -10, 10)
    x2 = binary_to_float(x2_bin, -10, 10)
    return x1, x2

# === Hitung Fitness ===
def calculate_fitness(chromosome):
    x1, x2 = decode_chromosome(chromosome)
    return -objective_function(x1, x2)  # Karena GA memaksimalkan fitness

# === Tournament Selection ===
def select_parent(population, fitnesses):
    participants = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    return max(participants, key=lambda x: x[1])[0]

# === Crossover Single Point ===
def crossover(parent1, parent2):
    if random.random() < PC:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

# === Mutasi Bit Flip ===
def mutate(chromosome):
    mutated = list(chromosome)
    for i in range(len(mutated)):
        if random.random() < PM:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)

# === Algoritma Genetika Utama ===
def genetic_algorithm():
    population = initialize_population()
    best_history = []

    for generation in range(MAX_GENERATIONS):
        fitnesses = [calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_chromosome = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_history.append(best_fitness)

        # Elitisme
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        new_population = [population[i] for i in elite_indices]

        # Buat populasi baru
        while len(new_population) < POP_SIZE:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]

        # Cetak info tiap 10 generasi
        if generation % 10 == 0 or generation == MAX_GENERATIONS - 1:
            x1, x2 = decode_chromosome(best_chromosome)
            print(f"Generasi {generation:3d} | f(x) terbaik = {-best_fitness:.6f} | x1 = {x1:.4f} | x2 = {x2:.4f}")

    # Output akhir
    best_fitnesses = [calculate_fitness(ind) for ind in population]
    best_idx = np.argmax(best_fitnesses)
    best_chromosome = population[best_idx]
    x1, x2 = decode_chromosome(best_chromosome)
    best_fx = -best_fitnesses[best_idx]

    print("\n=== HASIL AKHIR ===")
    print(f"Kromosom terbaik : {best_chromosome}")
    print(f"Nilai x1         : {x1:.6f}")
    print(f"Nilai x2         : {x2:.6f}")
    print(f"Minimum f(x1,x2) : {best_fx:.6f}")

    return best_chromosome, x1, x2, best_fx

# === Eksekusi Program ===
if __name__ == "__main__":
    genetic_algorithm()
