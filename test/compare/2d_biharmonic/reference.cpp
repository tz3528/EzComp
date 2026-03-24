//===-- reference.cpp -----------------------------------------*- C++ -*-===//
//
// 2D 双调和方程参考实现
// 方程: ∂u/∂t = -D·∇⁴u
// 其中 ∇⁴u = ∂⁴u/∂x⁴ + 2·∂⁴u/∂x²∂y² + ∂⁴u/∂y⁴
// 离散化: FDM (有限差分法), 13点模板
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <hdf5.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// 网格参数
constexpr int NX = 2001;     // x 方向点数
constexpr int NY = 2001;     // y 方向点数
constexpr int NT = 500;      // 时间步数

constexpr double X_LOWER = 0.0;
constexpr double X_UPPER = 2000.0;
constexpr double Y_LOWER = 0.0;
constexpr double Y_UPPER = 2000.0;

constexpr double D = 1e-6;   // 四阶扩散系数

// 格式化输出用时
static void print_timer(const char* label, const char* tag, long long ns) {
    auto hours = ns / 3600000000000LL;
    ns %= 3600000000000LL;
    auto minutes = ns / 60000000000LL;
    ns %= 60000000000LL;
    auto seconds = ns / 1000000000LL;
    ns %= 1000000000LL;
    auto milliseconds = ns / 1000000LL;

    if (tag && tag[0] != '\0') {
        std::fprintf(stderr, "[TIMER] [%s] %s: %lldh %lldm %llds %lldms\n",
                     tag, label,
                     (long long)hours, (long long)minutes,
                     (long long)seconds, (long long)milliseconds);
    } else {
        std::fprintf(stderr, "[TIMER] %s: %lldh %lldm %llds %lldms\n",
                     label,
                     (long long)hours, (long long)minutes,
                     (long long)seconds, (long long)milliseconds);
    }
}

// 写入 HDF5 文件
static bool write_hdf5(const char* output_file,
                       const std::vector<double>& u,
                       const std::vector<double>& coord_x,
                       const std::vector<double>& coord_y) {
    hid_t file = H5Fcreate(output_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return false;

    // 写入 result 数据集
    hsize_t dims[2] = { static_cast<hsize_t>(NX), static_cast<hsize_t>(NY) };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    if (space < 0) { H5Fclose(file); return false; }

    hid_t dset = H5Dcreate2(file, "/result", H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); H5Fclose(file); return false; }

    if (H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u.data()) < 0) {
        H5Dclose(dset); H5Sclose(space); H5Fclose(file); return false;
    }

    H5Dclose(dset);
    H5Sclose(space);

    // 写入 coord_x
    hsize_t dim_x[1] = { static_cast<hsize_t>(NX) };
    hid_t space_x = H5Screate_simple(1, dim_x, nullptr);
    hid_t dset_x = H5Dcreate2(file, "/coord_x", H5T_NATIVE_DOUBLE, space_x,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coord_x.data());
    H5Dclose(dset_x);
    H5Sclose(space_x);

    // 写入 coord_y
    hsize_t dim_y[1] = { static_cast<hsize_t>(NY) };
    hid_t space_y = H5Screate_simple(1, dim_y, nullptr);
    hid_t dset_y = H5Dcreate2(file, "/coord_y", H5T_NATIVE_DOUBLE, space_y,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_y, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coord_y.data());
    H5Dclose(dset_y);
    H5Sclose(space_y);

    H5Fclose(file);
    return true;
}

int main(int argc, char* argv[]) {
    using Clock = std::chrono::steady_clock;

    // 解析命令行参数
    const char* output_file = "expected_result.h5";
    const char* tag = "";

    if (argc >= 2) {
        output_file = argv[1];
    }
    if (argc >= 3) {
        tag = argv[2];
    }

    // ========== 开始计时 ==========
    auto t_compute_start = Clock::now();

    // 坐标数组
    std::vector<double> coord_x(NX);
    std::vector<double> coord_y(NY);

    double dx = (X_UPPER - X_LOWER) / (NX - 1);  // dx = 1.0
    double dy = (Y_UPPER - Y_LOWER) / (NY - 1);  // dy = 1.0
    double dt = 0.5;

    for (int i = 0; i < NX; ++i) {
        coord_x[i] = X_LOWER + i * dx;
    }
    for (int j = 0; j < NY; ++j) {
        coord_y[j] = Y_LOWER + j * dy;
    }

    // 场变量 (双缓冲)
    std::vector<double> u_curr(NX * NY, 0.0);
    std::vector<double> u_next(NX * NY, 0.0);

    // 初始条件: 中心高斯分布
    // u(x, y, 0) = 100 * exp(-((x-500)² + (y-500)²) / 5000)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            double x = coord_x[i];
            double y = coord_y[j];
            u_curr[i * NY + j] = 100.0 * std::exp(-((x - 500) * (x - 500) +
                                                     (y - 500) * (y - 500)) / 5000.0);
        }
    }

    // 边界条件函数（四阶方程需要两层边界）
    auto apply_boundary = [&]() {
        // 左边界 (i=0, i=1)
        for (int j = 0; j < NY; ++j) {
            u_curr[0 * NY + j] = 0.0;
            u_curr[1 * NY + j] = 0.0;
        }
        // 右边界 (i=NX-1, i=NX-2)
        for (int j = 0; j < NY; ++j) {
            u_curr[(NX - 1) * NY + j] = 0.0;
            u_curr[(NX - 2) * NY + j] = 0.0;
        }
        // 下边界 (j=0, j=1)
        for (int i = 0; i < NX; ++i) {
            u_curr[i * NY + 0] = 0.0;
            u_curr[i * NY + 1] = 0.0;
        }
        // 上边界 (j=NY-1, j=NY-2)
        for (int i = 0; i < NX; ++i) {
            u_curr[i * NY + (NY - 1)] = 0.0;
            u_curr[i * NY + (NY - 2)] = 0.0;
        }
    };

    // 应用初始边界条件
    apply_boundary();

    // 预计算系数
    double dx2 = dx * dx;       // dx²
    double dy2 = dy * dy;       // dy²
    double dx4 = dx2 * dx2;     // dx⁴
    double dy4 = dy2 * dy2;     // dy⁴
    double dx2dy2 = dx2 * dy2;  // dx²·dy²

    // ========== 时间迭代 ==========
    for (int t = 0; t < NT; ++t) {
        // 计算内部点 (需要两层边界，所以 i, j 从 2 开始，到 NX-3, NY-3 结束)
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 2; i < NX - 2; ++i) {
            for (int j = 2; j < NY - 2; ++j) {
                int idx = i * NY + j;

                // x 方向四阶导数: (u_{i-2} - 4u_{i-1} + 6u_i - 4u_{i+1} + u_{i+2}) / dx⁴
                double d4x = (u_curr[(i - 2) * NY + j] - 4.0 * u_curr[(i - 1) * NY + j]
                             + 6.0 * u_curr[idx]
                             - 4.0 * u_curr[(i + 1) * NY + j] + u_curr[(i + 2) * NY + j]) / dx4;

                // y 方向四阶导数: (u_{j-2} - 4u_{j-1} + 6u_j - 4u_{j+1} + u_{j+2}) / dy⁴
                double d4y = (u_curr[i * NY + (j - 2)] - 4.0 * u_curr[i * NY + (j - 1)]
                             + 6.0 * u_curr[idx]
                             - 4.0 * u_curr[i * NY + (j + 1)] + u_curr[i * NY + (j + 2)]) / dy4;

                // 混合四阶导数: ∂⁴u/∂x²∂y²
                // = [u_{i+1,j+1} + u_{i+1,j-1} + u_{i-1,j+1} + u_{i-1,j-1}
                //    - 2(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1})
                //    + 4u_{i,j}] / (dx²·dy²)
                double d4xy = (u_curr[(i + 1) * NY + (j + 1)] + u_curr[(i + 1) * NY + (j - 1)]
                              + u_curr[(i - 1) * NY + (j + 1)] + u_curr[(i - 1) * NY + (j - 1)]
                              - 2.0 * (u_curr[(i + 1) * NY + j] + u_curr[(i - 1) * NY + j]
                                      + u_curr[i * NY + (j + 1)] + u_curr[i * NY + (j - 1)])
                              + 4.0 * u_curr[idx]) / dx2dy2;

                // 完整双调和算子: ∇⁴u = ∂⁴u/∂x⁴ + 2·∂⁴u/∂x²∂y² + ∂⁴u/∂y⁴
                double bilaplacian = d4x + 2.0 * d4xy + d4y;

                // 时间推进: u^{n+1} = u^n - D·∇⁴u·Δt
                u_next[idx] = u_curr[idx] - D * bilaplacian * dt;
            }
        }

        // 交换缓冲区
        std::swap(u_curr, u_next);

        // 应用边界条件
        apply_boundary();
    }

    // ========== 计算完成，输出计时 ==========
    auto t_compute_end = Clock::now();
    auto compute_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_compute_end - t_compute_start).count();
    print_timer("计算用时", tag, compute_ns);

    // ========== 输出 HDF5 文件 ==========
    auto t_output_start = Clock::now();

    if (!write_hdf5(output_file, u_curr, coord_x, coord_y)) {
        std::fprintf(stderr, "Error: Failed to write HDF5 file\n");
        return 1;
    }

    auto t_output_end = Clock::now();
    auto output_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_output_end - t_output_start).count();
    print_timer("输出 HDF5 文件", tag, output_ns);

    std::fprintf(stderr, "结果已写入 %s\n", output_file);
    return 0;
}
