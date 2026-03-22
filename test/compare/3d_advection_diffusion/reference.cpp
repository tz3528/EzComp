//===-- reference.cpp -----------------------------------------*- C++ -*-===//
//
// 3D 对流扩散方程参考实现
// 方程: ∂u/∂t + vx*∂u/∂x + vy*∂u/∂y + vz*∂u/∂z = α*(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
// 离散化: FDM (对流项-迎风格式, 扩散项-中心差分)
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
constexpr int NX = 401;     // x 方向点数
constexpr int NY = 401;     // y 方向点数
constexpr int NZ = 401;     // z 方向点数
constexpr int NT = 100;     // 时间步数

constexpr double X_LOWER = 0.0;
constexpr double X_UPPER = 400.0;
constexpr double Y_LOWER = 0.0;
constexpr double Y_UPPER = 400.0;
constexpr double Z_LOWER = 0.0;
constexpr double Z_UPPER = 400.0;

// 物理参数
constexpr double ALPHA = 0.1;   // 扩散系数
constexpr double VX = 0.4;      // x方向对流速度
constexpr double VY = 0.3;      // y方向对流速度
constexpr double VZ = 0.2;      // z方向对流速度

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
                       const std::vector<double>& coord_y,
                       const std::vector<double>& coord_z) {
    hid_t file = H5Fcreate(output_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return false;

    // 写入 result 数据集
    hsize_t dims[3] = { static_cast<hsize_t>(NX), static_cast<hsize_t>(NY), static_cast<hsize_t>(NZ) };
    hid_t space = H5Screate_simple(3, dims, nullptr);
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

    // 写入 coord_z
    hsize_t dim_z[1] = { static_cast<hsize_t>(NZ) };
    hid_t space_z = H5Screate_simple(1, dim_z, nullptr);
    hid_t dset_z = H5Dcreate2(file, "/coord_z", H5T_NATIVE_DOUBLE, space_z,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_z, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coord_z.data());
    H5Dclose(dset_z);
    H5Sclose(space_z);

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
    std::vector<double> coord_z(NZ);

    double dx = (X_UPPER - X_LOWER) / (NX - 1);
    double dy = (Y_UPPER - Y_LOWER) / (NY - 1);
    double dz = (Z_UPPER - Z_LOWER) / (NZ - 1);
    double dt = 0.5;

    for (int i = 0; i < NX; ++i) {
        coord_x[i] = X_LOWER + i * dx;
    }
    for (int j = 0; j < NY; ++j) {
        coord_y[j] = Y_LOWER + j * dy;
    }
    for (int k = 0; k < NZ; ++k) {
        coord_z[k] = Z_LOWER + k * dz;
    }

    // 浓度场 (双缓冲)
    std::vector<double> u_curr(NX * NY * NZ, 0.0);
    std::vector<double> u_next(NX * NY * NZ, 0.0);

    // 初始条件: 中心高斯分布
    double x_center = 200.0;
    double y_center = 200.0;
    double z_center = 200.0;
    double sigma_sq = 10000.0;
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                double x = coord_x[i];
                double y = coord_y[j];
                double z = coord_z[k];
                u_curr[i * NY * NZ + j * NZ + k] = 100.0 * std::exp(
                    -((x - x_center) * (x - x_center) +
                      (y - y_center) * (y - y_center) +
                      (z - z_center) * (z - z_center)) / sigma_sq);
            }
        }
    }

    // 边界条件函数
    auto apply_boundary = [&]() {
        // 入流边界 (x=0, y=0, z=0): u = 0
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[0 * NY * NZ + j * NZ + k] = 0.0;
            }
        }
        for (int i = 0; i < NX; ++i) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[i * NY * NZ + 0 * NZ + k] = 0.0;
            }
        }
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                u_curr[i * NY * NZ + j * NZ + 0] = 0.0;
            }
        }
        // 出流边界 (x=400, y=400, z=400): 零梯度
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[(NX - 1) * NY * NZ + j * NZ + k] = u_curr[(NX - 2) * NY * NZ + j * NZ + k];
            }
        }
        for (int i = 0; i < NX; ++i) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[i * NY * NZ + (NY - 1) * NZ + k] = u_curr[i * NY * NZ + (NY - 2) * NZ + k];
            }
        }
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                u_curr[i * NY * NZ + j * NZ + (NZ - 1)] = u_curr[i * NY * NZ + j * NZ + (NZ - 2)];
            }
        }
    };

    // 应用初始边界条件
    apply_boundary();

    // 扩散系数
    double rx = ALPHA * dt / (dx * dx);
    double ry = ALPHA * dt / (dy * dy);
    double rz = ALPHA * dt / (dz * dz);
    // 对流系数 (迎风格式)
    double cx_neg = std::max(0.0, VX) * dt / dx;
    double cx_pos = std::max(0.0, -VX) * dt / dx;
    double cy_neg = std::max(0.0, VY) * dt / dy;
    double cy_pos = std::max(0.0, -VY) * dt / dy;
    double cz_neg = std::max(0.0, VZ) * dt / dz;
    double cz_pos = std::max(0.0, -VZ) * dt / dz;

    // ========== 时间迭代 ==========
    for (int t = 0; t < NT; ++t) {
        // 计算内部点
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static) collapse(3)
#endif
        for (int i = 1; i < NX - 1; ++i) {
            for (int j = 1; j < NY - 1; ++j) {
                for (int k = 1; k < NZ - 1; ++k) {
                    int idx = i * NY * NZ + j * NZ + k;
                    double u_center = u_curr[idx];
                    double u_left = u_curr[(i - 1) * NY * NZ + j * NZ + k];
                    double u_right = u_curr[(i + 1) * NY * NZ + j * NZ + k];
                    double u_front = u_curr[i * NY * NZ + (j - 1) * NZ + k];
                    double u_back = u_curr[i * NY * NZ + (j + 1) * NZ + k];
                    double u_bottom = u_curr[i * NY * NZ + j * NZ + (k - 1)];
                    double u_top = u_curr[i * NY * NZ + j * NZ + (k + 1)];

                    // 扩散项 (中心差分)
                    double diffusion = rx * (u_left + u_right - 2.0 * u_center)
                                     + ry * (u_front + u_back - 2.0 * u_center)
                                     + rz * (u_bottom + u_top - 2.0 * u_center);

                    // 对流项 (迎风格式)
                    double advection_x = cx_neg * (u_center - u_left)
                                       - cx_pos * (u_right - u_center);
                    double advection_y = cy_neg * (u_center - u_front)
                                       - cy_pos * (u_back - u_center);
                    double advection_z = cz_neg * (u_center - u_bottom)
                                       - cz_pos * (u_top - u_center);

                    u_next[idx] = u_center + diffusion - advection_x - advection_y - advection_z;
                }
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

    if (!write_hdf5(output_file, u_curr, coord_x, coord_y, coord_z)) {
        std::fprintf(stderr, "Error: Failed to write HDF5 file\n");
        return 1;
    }

    auto t_output_end = Clock::now();
    auto output_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_output_end - t_output_start).count();
    print_timer("输出 HDF5 文件", tag, output_ns);

    std::fprintf(stderr, "结果已写入 %s\n", output_file);
    return 0;
}
