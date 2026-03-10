#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <iomanip>

using namespace std;

const double EPS = 1e-5;
const int MAX_ITER = 10000;
const int N = 20000;

void init(vector<double>& A, vector<double>& b) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = N + 1.0;
    }
}

double solve_v1(const vector<double>& A, const vector<double>& b, int p) {
    vector<double> x(N, 0.0);
    vector<double> x_new(N, 0.0);
    int iter = 0;
    double criteria = 1.0;
    double tau = 1.0 / (N + 1.0);
    
    omp_set_num_threads(p);
    
    while (sqrt(criteria) > EPS && iter < MAX_ITER) {
        if (criteria > 1000) tau *= -1.0;
        criteria = 0.0;
        
        #pragma omp parallel for reduction(+ : criteria)
        for (int i = 0; i < N; i++) {
            double sigma = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i) sigma += A[i * N + j] * x[j];
            }
            x_new[i] = x[i] - tau * (sigma - b[i]);
            if (fabs(b[i]) > 1e-10) {
                double d = (sigma - b[i]) / b[i];
                criteria += d * d;
            } else {
                criteria += (sigma - b[i]) * (sigma - b[i]);
            }
        }
        
        #pragma omp parallel for
        for (int i = 0; i < N; i++) x[i] = x_new[i];
        iter++;
    }
    return sqrt(criteria);
}

double solve_v2(const vector<double>& A, const vector<double>& b, int p) {
    vector<double> x(N, 0.0);
    vector<double> x_new(N, 0.0);
    int iter = 0;
    double criteria = 1.0;
    double tau = 1.0 / (N + 1.0);
    bool should_stop = false;
    
    #pragma omp parallel num_threads(p)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int chunk = N / nthreads;
        int lb = tid * chunk;
        int ub = (tid == nthreads - 1) ? N : (lb + chunk);
        
        while (!should_stop) {
            #pragma omp single
            {
                if (sqrt(criteria) <= EPS || iter >= MAX_ITER) {
                    should_stop = true;
                } else {
                    if (criteria > 1000) tau *= -1.0;
                    criteria = 0.0;
                    iter++;
                }
            }
            
            #pragma omp barrier
            if (should_stop) break;
            
            double local_crit = 0.0;
            for (int i = lb; i < ub; i++) {
                double sigma = 0.0;
                for (int j = 0; j < N; j++) {
                    if (j != i) sigma += A[i * N + j] * x[j];
                }
                x_new[i] = x[i] - tau * (sigma - b[i]);
                if (fabs(b[i]) > 1e-10) {
                    double d = (sigma - b[i]) / b[i];
                    local_crit += d * d;
                } else {
                    local_crit += (sigma - b[i]) * (sigma - b[i]);
                }
            }
            
            #pragma omp atomic
            criteria += local_crit;
            #pragma omp barrier
            
            for (int i = lb; i < ub; i++) x[i] = x_new[i];
            #pragma omp barrier
        }
    }
    return sqrt(criteria);
}

int main() {
    int threads[] = {1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40};
    int n_threads = sizeof(threads) / sizeof(threads[0]);
    
    vector<double> A(N * N);
    vector<double> b(N);
    
    printf("Initializing %dx%d matrix...\n", N, N);
    init(A, b);
    printf("Done.\n\n");
    
    ofstream file("results_3.txt");
    file << fixed << setprecision(6);
    file << "# N = " << N << "\n";
    
    printf("=== VARIANT 1 ===\n");
    double t1 = solve_v1(A, b, 1);
    printf("p=1: %.4fs\n", t1);
    file << "# VARIANT 1\n# Threads Time Speedup\n";
    file << "1 " << t1 << " 1.0000\n";
    
    for (int i = 1; i < n_threads; i++) {
        double tp = solve_v1(A, b, threads[i]);
        double speedup = t1 / tp;
        printf("p=%2d: %.4fs (S=%.3f)\n", threads[i], tp, speedup);
        file << threads[i] << " " << tp << " " << speedup << "\n";
    }
    
    printf("\n=== VARIANT 2 ===\n");
    double t2 = solve_v2(A, b, 1);
    printf("p=1: %.4fs\n", t2);
    file << "\n# VARIANT 2\n# Threads Time Speedup\n";
    file << "1 " << t2 << " 1.0000\n";
    
    for (int i = 1; i < n_threads; i++) {
        double tp = solve_v2(A, b, threads[i]);
        double speedup = t2 / tp;
        printf("p=%2d: %.4fs (S=%.3f)\n", threads[i], tp, speedup);
        file << threads[i] << " " << tp << " " << speedup << "\n";
    }
    
    file.close();
    
    printf("\nData saved to speedup_data.txt\n");
    return 0;
}