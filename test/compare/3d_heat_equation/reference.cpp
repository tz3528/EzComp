//===-- reference.cpp -----------------------------------------*- C++ -*-===//
//
// 3D 热方程参考实现
// 方程: ∂u/∂t = α * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
// 离散化: FDM (有限差分法)
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
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

constexpr double ALPHA = 0.15;  // 热扩散系数

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
    // argv[1]: 输出文件名 (默认: expected_result.h5)
    // argv[2]: 运行标识标签 (默认: 空)
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
    double dt = 1.0;

    for (int i = 0; i < NX; ++i) {
        coord_x[i] = X_LOWER + i * dx;
    }
    for (int j = 0; j < NY; ++j) {
        coord_y[j] = Y_LOWER + j * dy;
    }
    for (int k = 0; k < NZ; ++k) {
        coord_z[k] = Z_LOWER + k * dz;
    }

    // 温度场 (双缓冲)
    std::vector<double> u_curr(NX * NY * NZ, 0.0);
    std::vector<double> u_next(NX * NY * NZ, 0.0);

    // 初始条件: u(x, y, z, 0) = 0
    for (int i = 0; i < NX * NY * NZ; ++i) {
        u_curr[i] = 0.0;
    }

    // 边界条件函数
    auto apply_boundary = [&]() {
        // u(0, y, z, t) = 100 (左面)
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[0 * NY * NZ + j * NZ + k] = 100.0;
            }
        }
        // u(400, y, z, t) = 0 (右面)
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[(NX - 1) * NY * NZ + j * NZ + k] = 0.0;
            }
        }
        // u(x, 0, z, t) = 100 (前面)
        for (int i = 0; i < NX; ++i) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[i * NY * NZ + 0 * NZ + k] = 100.0;
            }
        }
        // u(x, 400, z, t) = 0 (后面)
        for (int i = 0; i < NX; ++i) {
            for (int k = 0; k < NZ; ++k) {
                u_curr[i * NY * NZ + (NY - 1) * NZ + k] = 0.0;
            }
        }
        // u(x, y, 0, t) = 100 (底面)
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                u_curr[i * NY * NZ + j * NZ + 0] = 100.0;
            }
        }
        // u(x, y, 400, t) = 0 (顶面)
        for (int i = 0; i < NX; ++i) {
            for (int j = 0; j < NY; ++j) {
                u_curr[i * NY * NZ + j * NZ + (NZ - 1)] = 0.0;
            }
        }
    };

    // 应用初始边界条件
    apply_boundary();

    // 系数
    double rx = ALPHA * dt / (dx * dx);
    double ry = ALPHA * dt / (dy * dy);
    double rz = ALPHA * dt / (dz * dz);

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

                    // 二阶中心差分 (3D)
                    u_next[idx] = u_center + rx * (u_left + u_right - 2.0 * u_center)
                                        + ry * (u_front + u_back - 2.0 * u_center)
                                        + rz * (u_bottom + u_top - 2.0 * u_center);
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
