#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <cmath>

namespace py = pybind11;

inline double dist2d(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// cand:    (N, M, 2)
// valid:   (N, M)
// offsets: (N, M)
// smooth_w: 후보 index 급변 penalty
// offset_w: centerline에서 멀어지는 penalty
py::array_t<int> shortest_loop_dp(
    py::array_t<double> cand,
    py::array_t<bool> valid,
    py::array_t<double> offsets,
    double smooth_w,
    double offset_w
) {
    auto c = cand.unchecked<3>();
    auto v = valid.unchecked<2>();
    auto o = offsets.unchecked<2>();

    int N = c.shape(0);
    int M = c.shape(1);

    const double INF = std::numeric_limits<double>::max() / 4.0;

    std::vector<int> best_path(N, 0);
    double best_total = INF;

    for (int start = 0; start < M; start++) {
        if (!v(0, start)) continue;

        std::vector<std::vector<double>> dp(
            N, std::vector<double>(M, INF)
        );

        std::vector<std::vector<int>> parent(
            N, std::vector<int>(M, -1)
        );

        dp[0][start] = offset_w * o(0, start) * o(0, start);

        for (int i = 1; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (!v(i, j)) continue;

                double xj = c(i, j, 0);
                double yj = c(i, j, 1);

                for (int k = 0; k < M; k++) {
                    if (!v(i - 1, k)) continue;
                    if (dp[i - 1][k] >= INF) continue;

                    double xk = c(i - 1, k, 0);
                    double yk = c(i - 1, k, 1);

                    double d = dist2d(xj, yj, xk, yk);

                    double jump = static_cast<double>(j - k);
                    double smooth_cost = smooth_w * jump * jump;

                    double off = o(i, j);
                    double offset_cost = offset_w * off * off;

                    double cost = d + smooth_cost + offset_cost;
                    double total = dp[i - 1][k] + cost;

                    if (total < dp[i][j]) {
                        dp[i][j] = total;
                        parent[i][j] = k;
                    }
                }
            }
        }

        for (int end = 0; end < M; end++) {
            if (!v(N - 1, end)) continue;
            if (dp[N - 1][end] >= INF) continue;

            double close_d = dist2d(
                c(N - 1, end, 0),
                c(N - 1, end, 1),
                c(0, start, 0),
                c(0, start, 1)
            );

            double close_jump = static_cast<double>(end - start);
            double close_smooth = smooth_w * close_jump * close_jump;

            double total = dp[N - 1][end] + close_d + close_smooth;

            if (total < best_total) {
                std::vector<int> path(N, 0);
                path[N - 1] = end;

                bool ok = true;

                for (int i = N - 1; i > 0; i--) {
                    int prev = parent[i][path[i]];
                    if (prev < 0) {
                        ok = false;
                        break;
                    }
                    path[i - 1] = prev;
                }

                if (ok) {
                    best_total = total;
                    best_path = path;
                }
            }
        }
    }

    auto result = py::array_t<int>(N);
    auto r = result.mutable_unchecked<1>();

    for (int i = 0; i < N; i++) {
        r(i) = best_path[i];
    }

    return result;
}

PYBIND11_MODULE(raceline_core, m) {
    m.def("shortest_loop_dp", &shortest_loop_dp);
}