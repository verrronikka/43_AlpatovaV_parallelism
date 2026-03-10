#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x)
{
    return exp(-x * x);
}

double integrate(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int p)
{
    double h = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel num_threads(p)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
        
        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

double run_serial()
{
    double t = omp_get_wtime();
    double res = integrate(a, b, nsteps);
    t = omp_get_wtime() - t;
    // printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel(int p)
{
    double t = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps, p);
    t = omp_get_wtime() - t;
    // printf("Result (parallel, %d threads): %.12f; error %.12f\n", p, res, fabs(res - sqrt(PI)));
    return t;
}

int main()
{
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_tests = sizeof(threads) / sizeof(threads[0]);
    
    printf("threads,time,speedup,error\n");
    
    double t_serial = run_serial();
    
    for (int i = 0; i < num_tests; i++)
    {
        int p = threads[i];
        
        if (p > omp_get_max_threads())
        {
            printf("%d,SKIP,SKIP,SKIP\n", p);
            continue;
        }
        
        double t_parallel = run_parallel(p);
        double speedup = t_serial / t_parallel;
        double error = fabs(integrate_omp(func, a, b, nsteps, p) - sqrt(PI));
        
        printf("%d,%.6f,%.6f,%.15f\n", p, t_parallel, speedup, error);
    }
    
    return 0;
}