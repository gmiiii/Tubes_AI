# =============================================================
#             ALGORITMA GENETIKA UNTUK MINIMISASI f(x1, x2)
# =============================================================
# Tujuan: Mencari nilai minimum dari fungsi non-linear kompleks
# dengan representasi kromosom biner tanpa menggunakan library eksternal.
# =============================================================

# -------------------------------------------------------------
# BAGIAN 1: Fungsi Matematika Dasar (tanpa math atau numpy)
# -------------------------------------------------------------
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def power(x, n):
    result = 1.0
    for _ in range(n):
        result *= x
    return result

def sin(x, terms=10):
    x = x % (2 * 3.1415926535)  # Normalisasi sudut
    result = 0.0
    for n in range(terms):
        sign = -1 if n % 2 else 1
        result += sign * power(x, 2 * n + 1) / factorial(2 * n + 1)
    return result

def cos(x, terms=10):
    x = x % (2 * 3.1415926535)
    result = 0.0
    for n in range(terms):
        sign = -1 if n % 2 else 1
        result += sign * power(x, 2 * n) / factorial(2 * n)
    return result

def tan(x):
    c = cos(x)
    if abs(c) < 1e-8:  # Hindari pembagian dengan nol
        return float('inf')
    return sin(x) / c

def sqrt(x, iterations=10):
    if x < 0:
        return float('nan')
    guess = x / 2.0
    for _ in range(iterations):
        guess = (guess + x / guess) / 2.0
    return guess

def exp(x, terms=20):
    result = 1.0
    for i in range(1, terms):
        result += power(x, i) / factorial(i)
    return result

# -------------------------------------------------------------
# BAGIAN 2: Random Number Generator (tanpa import random)
# -------------------------------------------------------------
seed = 123456789  # Seed acak awal
def rand():
    global seed
    seed = (1103515245 * seed + 12345) % (2 ** 31)
    return seed / (2 ** 31)

def randint(a, b):
    return a + int(rand() * (b - a + 1))

def random_bit():
    return '1' if rand() < 0.5 else '0'

# -------------------------------------------------------------
# BAGIAN 3: Parameter GA
# -------------------------------------------------------------
POP_SIZE = 10             # Ukuran populasi
GEN_LENGTH = 20           # Panjang bitstring kromosom
CHROMOSOME_BITS = 10      # Panjang representasi tiap variabel (x1 dan x2)
X_MIN, X_MAX = -10, 10    # Batas domain variabel
MAX_GENERATIONS = 10      # Jumlah generasi (loop)
PC = 0.8                  # Probabilitas crossover
PM = 0.01                 # Probabilitas mutasi per-bit

# -------------------------------------------------------------
# BAGIAN 4: Representasi Kromosom & Fitness
# -------------------------------------------------------------
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

def f(x1, x2):
    try:
        return -(sin(x1) * cos(x2) * tan(x1 + x2) + 0.75 * exp(1 - sqrt(x1 * x1)))
    except:
        return float('inf')

def fitness(bitstring):
    x1, x2 = decode(bitstring)
    val = f(x1, x2)
    if val == float('inf'):
        return 0.0001  # Hindari divide by zero
    return 1 / (1 + val)

# -------------------------------------------------------------
# BAGIAN 5: Operasi Genetika (GA Core)
# -------------------------------------------------------------
def select(population):
    # Tournament Selection (2-individu)
    i, j = randint(0, len(population) - 1), randint(0, len(population) - 1)
    return population[i] if fitness(population[i]) > fitness(population[j]) else population[j]

def crossover(p1, p2):
    if rand() < PC:
        point = randint(1, GEN_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutate(bitstring):
    return ''.join((bit if rand() > PM else ('0' if bit == '1' else '1')) for bit in bitstring)

# -------------------------------------------------------------
# BAGIAN 6: Proses Evolusi Generasi ke Generasi
# -------------------------------------------------------------
def run_ga():
    population = initialize_population()
    best = population[0]

    for gen in range(MAX_GENERATIONS):
        new_population = []

        # Elitisme: simpan individu terbaik
        elite = max(population, key=fitness)
        new_population.append(elite)

        while len(new_population) < POP_SIZE:
            parent1 = select(population)
            parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(child2))

        population = new_population

        # Perbarui solusi terbaik jika ada
        if fitness(elite) > fitness(best):
            best = elite

        # Cetak log tiap generasi
        x1, x2 = decode(elite)
        print("Generasi", gen + 1, ": Best Fitness =", fitness(elite), "| x1 =", x1, "| x2 =", x2)

    # Cetak hasil akhir
    x1, x2 = decode(best)
    print("\n=== HASIL AKHIR ===")
    print("Kromosom terbaik:", best)
    print("x1 =", x1)
    print("x2 =", x2)
    print("f(x1, x2) =", f(x1, x2))

# -------------------------------------------------------------
# BAGIAN 7: Eksekusi
# -------------------------------------------------------------
if __name__ == "__main__":
    run_ga()
