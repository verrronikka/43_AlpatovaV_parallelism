#include <iostream>
#include <thread>
#include <vector>
#include <iomanip>
#include <chrono>
#include <algorithm>

double wtime() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void init_data_parallel(std::vector<double>& a, std::vector<double>& b, 
                        int m, int n, int num_threads) {
    std::vector<std::thread> threads;
    
    auto init_matrix = [&](int tid, int total_threads) {
        int rows_per_thread = (m + total_threads - 1) / total_threads;
        int start_row = std::min(tid * rows_per_thread, m);
        int end_row = std::min(start_row + rows_per_thread, m);
        
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < n; ++j) {
                a[i * n + j] = (i == j) ? 2.0 : 1.0;
            }
        }
    };
    
    for (int t = 0; t < num_threads; ++t)
        threads.emplace_back(init_matrix, t, num_threads);
    for (auto& th : threads) th.join();
    threads.clear();
    
    auto init_vector = [&](int tid, int total_threads) {
        int chunk = (n + total_threads - 1) / total_threads;
        int start = std::min(tid * chunk, n);
        int end = std::min(start + chunk, n);
        for (int i = start; i < end; ++i) b[i] = 1.0;
    };
    
    for (int t = 0; t < num_threads; ++t)
        threads.emplace_back(init_vector, t, num_threads);
    for (auto& th : threads) th.join();
}

void matrix_vector_product_parallel(std::vector<double>& a, 
                                    const std::vector<double>& b,
                                    std::vector<double>& c, 
                                    int m, int n, int num_threads) {
    std::vector<std::thread> threads;
    
    auto compute_rows = [&](int tid, int total_threads) {
        int rows_per_thread = (m + total_threads - 1) / total_threads;
        int start_row = std::min(tid * rows_per_thread, m);
        int end_row = std::min(start_row + rows_per_thread, m);
        
        for (int i = start_row; i < end_row; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += a[i * n + j] * b[j];
            }
            c[i] = sum;
        }
    };
    
    for (int t = 0; t < num_threads; ++t)
        threads.emplace_back(compute_rows, t, num_threads);
    for (auto& th : threads) th.join();
}

int main() {
    std::vector<int> sizes = {20000, 40000};
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40};
    
    std::cout << std::fixed << std::setprecision(4);
    
    for (int size : sizes) {
        std::cout << "# Matrix size: " << size << "x" << size << "\n";
        
        for (int p : thread_counts) {
            std::vector<double> a(size * size);
            std::vector<double> b(size);
            std::vector<double> c(size);
            
            init_data_parallel(a, b, size, size, p);
            
            double t_start = wtime();
            matrix_vector_product_parallel(a, b, c, size, size, p);
            double t_end = wtime();
            
            std::cout << "p=" << std::setw(2) << p << ": " << (t_end - t_start) << "s\n";
        }
        
        std::cout << "\n";
    }
    
    return 0;
}