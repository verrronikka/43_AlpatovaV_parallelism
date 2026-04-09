#include <stdio.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iomanip>

const int N = 25000;
const double TOL = 1e-6;
const int MAX_ITER = 10000;

void init(std::vector<double>& A, std::vector<double>& b) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = N + 1.0;
    }
}

double norm_diff(const std::vector<double>& x, const std::vector<double>& x_new) {
    double max_diff = 0.0;

    #pragma omp parallel for reduction(max : max_diff)
    for (int i = 0; i < N; i++) {
        double d = fabs(x[i] - x_new[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}


// Два отдельных #pragma omp parallel for
double solve_v1(const std::vector<double>& A, const std::vector<double>& b, int threads) {
    std::vector<double> x(N, 0.0);
    std::vector<double> x_new(N, 0.0);

    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();

    int iter = 0;
    for (iter = 0; iter < MAX_ITER; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++)
                if (i != j) sum += A[i * N + j] * x[j];
            x_new[i] = (b[i] - sum) / 2.0;
        }

        #pragma omp parallel for
        for (int i = 0; i < N; i++)
            x[i] = x_new[i];

        if (norm_diff(x, x_new) < TOL) break;
    }

    double t1 = omp_get_wtime();
    printf("V1 finished after %d iterations\n", iter);
    return t1 - t0;
}

// Одна параллельная секция + #pragma omp for + барьеры
double solve_v2(const std::vector<double>& A, const std::vector<double>& b, int threads) {
    std::vector<double> x(N, 0.0);
    std::vector<double> x_new(N, 0.0);

    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();

    int iter = 0;
    bool converged = false;

    #pragma omp parallel shared(converged, iter) num_threads(threads)
    {
        while (!converged && iter < MAX_ITER) {
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++)
                    if (i != j) sum += A[i * N + j] * x[j];
                x_new[i] = (b[i] - sum) / 2.0;
            }

            // Копирование x_new -> x
            #pragma omp for
            for (int i = 0; i < N; i++)
                x[i] = x_new[i];

            #pragma omp single
            {
                iter++;
                if (norm_diff(x, x_new) < TOL) {
                    converged = true;
                }
            }

            #pragma omp barrier
        }
    }

    double t1 = omp_get_wtime();
    printf("V2 finished after %d iterations\n", iter);
    return t1 - t0;
}

int main() {
    int threads[] = {1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40};
    int n_threads = sizeof(threads) / sizeof(threads[0]);

    std::vector<double> A(static_cast<size_t>(N) * N);
    std::vector<double> b(N);

    printf("Initializing %dx%d matrix... (this will take some time)\n", N, N);
    init(A, b);
    printf("Initialization done.\n\n");

    std::ofstream file("results_3.txt");
    file << std::fixed << std::setprecision(6);
    file << "# N = " << N << "\n";

    // === VARIANT 1 ===
    printf("=== VERSION 1 ===\n");
    double t1 = solve_v1(A, b, 1);
    printf("p=1: %.4f s\n", t1);
    file << "# Version 1\n# p Time Speedup\n";
    file << "1 " << t1 << " 1.000000\n";

    for (int i = 1; i < n_threads; i++) {
        double tp = solve_v1(A, b, threads[i]);
        double speedup = t1 / tp;
        printf("p=%2d: %.4f s (S=%.3f)\n", threads[i], tp, speedup);
        file << threads[i] << " " << tp << " " << speedup << "\n";
    }

    // === VARIANT 2 ===
    printf("\n=== VERSION 2 ===\n");
    double t2 = solve_v2(A, b, 1);
    printf("p=1: %.4f s\n", t2);
    file << "\n# Version 2\n# p Time Speedup\n";
    file << "1 " << t2 << " 1.000000\n";

    for (int i = 1; i < n_threads; i++) {
        double tp = solve_v2(A, b, threads[i]);
        double speedup = t2 / tp;
        printf("p=%2d: %.4f s (S=%.3f)\n", threads[i], tp, speedup);
        file << threads[i] << " " << tp << " " << speedup << "\n";
    }

    file.close();
    printf("\nResults saved to results_3.txt\n");
    return 0;
}