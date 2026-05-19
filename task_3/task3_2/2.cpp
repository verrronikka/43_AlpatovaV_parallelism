#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <functional>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <string>
#include <iomanip>

template <typename T>
class TaskServer {
public:
    using TaskFunc = std::function<T()>;

    void start() {
        worker_ = std::jthread([this](std::stop_token stoken) { process(stoken); });
    }

    void stop() {
        worker_.request_stop();
        queue_cv_.notify_all();
    }

    size_t add_task(TaskFunc task) {
        std::lock_guard<std::mutex> lock(q_mtx_);
        size_t id = ++counter_;
        tasks_.emplace(id, std::move(task));
        queue_cv_.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(res_mtx_);
        res_cv_.wait(lock, [this, id] { return results_.count(id); });
        T val = results_[id];
        results_.erase(id);
        return val;
    }

private:
    void process(std::stop_token stoken) {
        while (true) {
            std::unique_lock<std::mutex> lock(q_mtx_);
            queue_cv_.wait(lock, [&stoken, this] {
                return !tasks_.empty() || stoken.stop_requested();
            });

            if (stoken.stop_requested() && tasks_.empty()) break;

            auto [id, task] = std::move(tasks_.front());
            tasks_.pop();
            lock.unlock();

            T result = task();

            std::lock_guard<std::mutex> r_lock(res_mtx_);
            results_[id] = result;
            res_cv_.notify_all();
        }
    }

    std::jthread worker_;
    size_t counter_{0};
    std::queue<std::pair<size_t, TaskFunc>> tasks_;
    std::mutex q_mtx_;
    std::condition_variable queue_cv_;
    std::unordered_map<size_t, T> results_;
    std::mutex res_mtx_;
    std::condition_variable res_cv_;
};

void run_client(int id, int n, TaskServer<double>& srv, const std::string& file) {
    std::vector<size_t> ids;
    std::vector<std::pair<double, double>> args;
    ids.reserve(n); args.reserve(n);

    std::mt19937 gen(id * 777 + 42);
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    std::function<double(double, double)> func;
    if (id == 1) func = [](double x, double) { return std::sin(x); };
    else if (id == 2) func = [](double x, double) { return std::sqrt(x); };
    else func = [](double x, double y) { return std::pow(x, y); };

    for (int i = 0; i < n; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        args.emplace_back(x, y);
        ids.push_back(srv.add_task([func, x, y]() { return func(x, y); }));
    }

    std::ofstream out(file);
    out << std::fixed << std::setprecision(15);
    for (size_t i = 0; i < ids.size(); ++i) {
        double res = srv.request_result(ids[i]);
        out << ids[i] << " " << args[i].first << " " << args[i].second << " " << res << "\n";
    }
    out.close();
    std::cout << "[Client " << id << "] Done.\n";
}

bool verify(const std::string& file, int id) {
    std::ifstream in(file);
    if (!in.is_open()) return false;

    std::function<double(double, double)> func;
    if (id == 1) func = [](double x, double) { return std::sin(x); };
    else if (id == 2) func = [](double x, double) { return std::sqrt(x); };
    else func = [](double x, double y) { return std::pow(x, y); };

    size_t task_id;
    double x, y, res;
    while (in >> task_id >> x >> y >> res) {
        if (std::abs(func(x, y) - res) > 1e-9) return false;
    }
    return true;
}

int main() {
    const int N = 100;
    TaskServer<double> server;

    server.start();

    std::thread c1(run_client, 1, N, std::ref(server), "c1_sin.txt");
    std::thread c2(run_client, 2, N, std::ref(server), "c2_sqrt.txt");
    std::thread c3(run_client, 3, N, std::ref(server), "c3_pow.txt");

    c1.join(); c2.join(); c3.join();
    server.stop();

    bool ok = verify("c1_sin.txt", 1) && verify("c2_sqrt.txt", 2) && verify("c3_pow.txt", 3);
    std::cout << (ok ? "✅ Tests passed." : "❌ Tests failed.") << "\n";
    return ok ? 0 : 1;
}