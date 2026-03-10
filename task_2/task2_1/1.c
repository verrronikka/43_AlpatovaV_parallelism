#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

double run_serial(int m, int n)
{
    double *a, *b, *c;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = calloc(sizeof(*c), m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    t = omp_get_wtime() - t;
    
    // printf("Elapsed time (serial): %.6f sec.\n", t);
    
    free(a);
    free(b);
    free(c);
    return t;
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int p)
{
    #pragma omp parallel num_threads(p)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

double run_parallel(int m, int n, int p)
{
    double *a, *b, *c;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = calloc(sizeof(*c), m);

    #pragma omp parallel num_threads(p)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        }
    }

    #pragma omp parallel for num_threads(p)
    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n, p);
    t = omp_get_wtime() - t;
    
    // printf("Elapsed time (parallel): %.6f sec.\n", t);
    
    free(a);
    free(b);
    free(c);
    return t;
}

int main()
{

    printf("DEBUG: Max threads available: %d\n", omp_get_max_threads());
    int sizes[] = {20000, 40000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_threads = sizeof(threads) / sizeof(threads[0]);
    

    printf("matrix_size,threads,time,speedup\n");
    

    for (int s = 0; s < num_sizes; s++) {
        int m = sizes[s];
        int n = m;
        
        double t_serial = run_serial(m, n);
        
        for (int i = 0; i < num_threads; i++) {
            int p = threads[i];
            
            if (p > omp_get_max_threads()) {
                printf("%d,%d,SKIP,SKIP\n", m, p);
                continue;
            }
            
            double t_parallel = run_parallel(m, n, p);
            
            double speedup = t_serial / t_parallel;
            
            printf("%d,%d,%.6f,%.6f\n", m, p, t_parallel, speedup);
        }
    }

    return 0;
}