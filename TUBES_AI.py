# =============================================================
#   ALGORITMA GENETIKA UNTUK MINIMISASI f(x1, x2) (dengan tracing)
# =============================================================

# ------------------------ RANDOM ----------------------------
seed = 123456756
def rand():
    global seed
    seed = (1103515245 * seed + 12345) % (2 ** 31)
    return seed / (2 ** 31)

def randint(a, b):
    return a + int(rand() * (b - a + 1))

def random_bit():
    return '1' if rand() < 0.5 else '0'

# ---------------------- PARAMETER --------------------------
POP_SIZE = 5
GEN_LENGTH = 20
CHROMOSOME_BITS = 10
X_MIN, X_MAX = -10, 10
MAX_GENERATIONS = 5
PC = 0.8
PM = 0.01

# ---------------------- MATH FUNGSI -------------------------
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def power(x, n):
    result = 1.0
    for _ in range(n): result *= x
    return result

def sin(x, terms=10):
    x = x % (2 * 3.1415926535)
    return sum(((-1) ** n) * power(x, 2 * n + 1) / factorial(2 * n + 1) for n in range(terms))

def cos(x, terms=10):
    x = x % (2 * 3.1415926535)
    return sum(((-1) ** n) * power(x, 2 * n) / factorial(2 * n) for n in range(terms))

def tan(x):
    c = cos(x)
    return float('inf') if abs(c) < 1e-8 else sin(x) / c

def sqrt(x, iterations=10):
    guess = x / 2.0
    for _ in range(iterations):
        guess = (guess + x / guess) / 2.0
    return guess

def exp(x, terms=20):
    return sum(power(x, i) / factorial(i) for i in range(terms))

# --------------------- GA UTILITAS --------------------------
def f(x1, x2):
    try:
        return -(sin(x1) * cos(x2) * tan(x1 + x2) + 0.75 * exp(1 - sqrt(x1 * x1)))
    except:
        return float('inf')

def fitness(bitstring):
    x1, x2 = decode(bitstring)
    val = f(x1, x2)
    return 0.0001 if val == float('inf') else 1 / (1 + val)

def random_bitstring(length):
    return ''.join(random_bit() for _ in range(length))

def initialize_population():
    return [random_bitstring(GEN_LENGTH) for _ in range(POP_SIZE)]

def binary_to_real(bits, min_val, max_val):
    integer = int(bits, 2)
    max_int = 2 ** CHROMOSOME_BITS - 1
    return min_val + (max_val - min_val) * integer / max_int

def decode(bitstring):
    x1_bits = bitstring[:CHROMOSOME_BITS]
    x2_bits = bitstring[CHROMOSOME_BITS:]
    return binary_to_real(x1_bits, X_MIN, X_MAX), binary_to_real(x2_bits, X_MIN, X_MAX)

def select(population):
    i, j = randint(0, len(population) - 1), randint(0, len(population) - 1)
    return population[i] if fitness(population[i]) > fitness(population[j]) else population[j]

def crossover(p1, p2):
    if rand() < PC:
        point = randint(1, GEN_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutate(bitstring):
    return ''.join(bit if rand() > PM else ('0' if bit == '1' else '1') for bit in bitstring)

# ------------------ MAIN GA DENGAN TRACING -------------------
def run_ga_verbose_trace():
    population = initialize_population()
    best = population[0]

    for gen in range(MAX_GENERATIONS):
        print(f"\n=== Generasi {gen + 1} ===")
        new_population = []

        # Elitisme
        elite = max(population, key=fitness)
        new_population.append(elite)

        trace = []
        while len(new_population) < POP_SIZE:
            parent1 = select(population)
            parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            child1_mut = mutate(child1)
            child2_mut = mutate(child2)
            new_population.append(child1_mut)
            if len(new_population) < POP_SIZE:
                new_population.append(child2_mut)
            trace.append((parent1, parent2, child1_mut, child2_mut))

        # Cetak hasil semua individu
        all_individuals = [elite] + [c for t in trace for c in t[2:]]
        fitnesses = [fitness(ind) for ind in all_individuals]
        best_idx = fitnesses.index(max(fitnesses))

        for idx, (ind, fit) in enumerate(zip(all_individuals, fitnesses)):
            x1, x2 = decode(ind)
            tag = " <- BEST" if idx == best_idx else ""
            print(f"{idx + 1:2d}. {ind} | Fitness: {fit:.4f} | x1: {x1:.2f}, x2: {x2:.2f}{tag}")

        population = new_population
        best = max(population, key=fitness)

    # Final result
    x1, x2 = decode(best)
    print("\n=== HASIL AKHIR ===")
    print(f"Kromosom terbaik: {best}")
    print(f"x1 = {x1:.4f}, x2 = {x2:.4f}")
    print(f"f(x1, x2) = {f(x1, x2):.4f}")

# Jalankan program
if __name__ == "__main__":
    run_ga_verbose_trace()
