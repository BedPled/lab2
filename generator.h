#include <vector>
#include <thread>

#define A UINT64_C(6364136223846793005)
#define B 1
#define C 4294967296

unsigned get_num_threads();

std::vector<unsigned> pow_A(unsigned T) {
    std::vector<unsigned> result;
	result.reserve(T);
    result.emplace_back(A);
    for (unsigned i = 1; i < T + 1; i++) {
		unsigned next_A = (result[i-1] * A) % C;
        result.emplace_back(next_A);
    }
    return result;
}

#ifdef _MSC_VER
constexpr std::size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
#define CACHE_LINE 64
#endif

typedef struct element_t_
{
	alignas(CACHE_LINE) double value;
} element_t;


void srand(int seed);
typedef uint64_t word;
word rand(word s);
////
//# define A UINT64_C(6364136223846793005)
//# define B 1
//// c = 1 << 64

double randomizeSimple(unsigned* V, unsigned n, unsigned a, unsigned b) {
    word x = C;
    for (size_t i = 0; i < n; ++i) {
        x = A * x + B;
        V[i] = a + int(x % (b - a + 1));
    }

    int sum = 0;

    for (unsigned i = 0; i < n; ++i)
        sum += V[i];
    return sum / n;
}

struct LutRow {
    size_t a;
    size_t b;
};

std::unique_ptr<LutRow[]> get_lut(unsigned T) {
    auto lut = std::make_unique<LutRow[]>(T + 1);
    lut[0].a = A;
    lut[0].b = B;
    for (auto i = 1; i < T; ++i) {
        lut[i].a = lut[i - 1].a * A;
        lut[i].b = A * lut[i - 1].b + B;
    }
    return lut;
}

double  randomizeOMP(unsigned* V, unsigned n, unsigned a, unsigned b) {
    word S0 = C;
    size_t T = omp_get_num_procs();
    static auto lut = get_lut(T);

#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        size_t St = lut[t].a * S0 + lut[t].b;

        for (unsigned k = t; k < n; k += T) {
            V[k] = a + int(St % (b - a + 1));
            St = lut[t].a * St + lut[t].b;
        }
    }
    int sum = 0;
    for (unsigned i = 0; i < n; ++i)
        sum += V[i];
    return (sum / n);
}


double randomizeCPP(unsigned* V, unsigned N, unsigned min, unsigned max) {

    unsigned T = get_num_threads();
    std::vector<unsigned> multipliers = pow_A(T);
    double sum = 0;
    std::vector<element_t> partial(T, element_t{0.0});
    std::vector<std::thread> threads;
    unsigned seed = std::time(0);
    for (std::size_t t = 0; t < T; ++t)
        threads.emplace_back([t, T, V, N, seed, &partial, multipliers, min, max]() {
        auto At = multipliers.back();
        unsigned off = (B * (At - 1) / (A - 1)) % C;
            unsigned x = ((seed * multipliers[t]) % C + (B % C * (multipliers[t] - 1) / (A - 1)) % C) % C;
            double acc = 0;
            for (size_t i = t; i < N; i += T) {
                V[i] = x % (max - min) + min;
                acc += V[i];
                x = (x * At) % C + off % C;
            }
            partial[t].value = acc;
            });
    for (auto& thread:threads)
        thread.join();
    for (unsigned i = 0; i < T; ++i)
        sum += partial[i].value;
    return   (int) sum / N;
}