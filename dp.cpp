// Grupo de Otimizacao em Sistemas
// Univerisidade Federal de Santa Catarina

#include <algorithm>
#include <future>
#include <mutex>
#include <omp.h>
#include <vector>

using namespace std;
std::mutex database_mutex;

/**
 * Calculates the pricing using dynamic programming algorithm.
 *
 * @param job The job index.
 * @param JOBS The total number of jobs.
 * @param COVERS The total number of covers.
 * @param UPPER_K The upper limit for K.
 * @param shadow_price The array of shadow prices.
 * @param allow The array of allow values.
 * @param qtde_cortes The number of cuts.
 * @param T The total time.
 * @param priority The array of priorities.
 * @param uso_p The array of usage percentages.
 * @param min_statup The array of minimum startup times.
 * @param max_statup The array of maximum startup times.
 * @param min_cpu_time The array of minimum CPU times.
 * @param max_cpu_time The array of maximum CPU times.
 * @param min_periodo_job The array of minimum job periods.
 * @param max_periodo_job The array of maximum job periods.
 * @param win_min The array of minimum window times.
 * @param win_max The array of maximum window times.
 * @return The final column of the dynamic programming algorithm.
 */
extern "C" double *
pricing_dp(const int job, const int JOBS, const int COVERS, const int UPPER_K,
           const double shadow_price[], const double allow[], int qtde_cortes,
           const int T, const int priority[], const double uso_p[],
           const int min_statup[], const int max_statup[],
           const int min_cpu_time[], const int max_cpu_time[],
           const int min_periodo_job[], const int max_periodo_job[],
           const int win_min[], const int win_max[]) {

  const int y_min = min_statup[job];
  const int y_max = max_statup[job];

  const int t_min = min_cpu_time[job];
  const int t_max = max_cpu_time[job];

  const int p_min = min_periodo_job[job];
  const int p_max = max_periodo_job[job];

  const int w_min = win_min[job];
  const int w_max = win_max[job];

  int K = max(max_statup[job] - min_statup[job], 1) *
          max(max_cpu_time[job] - min_cpu_time[job], 1) *
          max(max_periodo_job[job] - min_periodo_job[job], 1);
  if (K > UPPER_K) {
    K = UPPER_K;
  }
  std::vector<int> allow0(T + 2, 1);
  std::vector<int> allow1(T + 2, 1);

  double c0 = -shadow_price[job];
  std::vector<double> c(T + 2, 0);

  for (int i = 0; i < T; i++) {
    c[i + 1] = shadow_price[i + JOBS + job * T];
  }

  if (qtde_cortes >= 1) {
    // #pragma omp parallel for
    for (int i = 0; i < qtde_cortes; i++) {
      if (allow[3 * i] == job) {

        int tempo = allow[3 * i + 1];

        // #pragma omp critical
        if (allow[3 * i + 2] == 0) {
          allow0[tempo + 1] = 1;
          allow1[tempo + 1] = 0;
        } else {
          allow0[tempo + 1] = 0;
          allow1[tempo + 1] = 1;
        }
      }
    }
  }

  // Dynamic Programming Algorithm

  // Declare the F and X arrays using std::vector
  std::vector<std::vector<std::vector<std::vector<double>>>> F(
      T + 2,
      std::vector<std::vector<std::vector<double>>>(
          t_max + 3, std::vector<std::vector<double>>(
                         y_max + 3, std::vector<double>(p_max + 3, 0.0))));

  std::vector<std::vector<std::vector<std::vector<int>>>> X(
      T + 1, std::vector<std::vector<std::vector<int>>>(
                 t_max + 3, std::vector<std::vector<int>>(
                                y_max + 3, std::vector<int>(p_max + 3, 0))));

  // Initialize the terminal conditions for the dynamic programming algorithm
  for (int z = 0; z <= t_max + 1; z++) {
    for (int w = 0; w <= y_max + 1; w++) {
      for (int p = 0; p <= p_max + 1; p++) {
        if (z <= t_max && w >= y_min && w <= y_max && p <= p_max) {
          F[T + 1][z + 1][w + 1][p + 1] = c0;
        } else {
          F[T + 1][z + 1][w + 1][p + 1] = std::numeric_limits<int>::min();
        }
      }
    }
  }

  for (int t = T; t >= 1; t--) {
// #pragma omp target map(tofrom:X) map(tofrom:F)
#pragma omp parallel for
    for (int z = 0; z <= t_max + 1; z++) {
      for (int w = 0; w <= y_max + 1; w++) {
        for (int p = 0; p <= p_max + 1; p++) {

          double cs = std::numeric_limits<int>::min();
          int xs = -1;
          // int t_index = omp_get_thread_num() % 2;

          // [x_t=0] Consider not running the job at time t
          if ((z == 0 && p < p_max && allow0[t] == 1) ||
              (z > 0 && z >= t_min && z <= t_max && p < p_max &&
               allow0[t] == 1)) {
            if (F[(t + 1)][0 + 1][w + 1][p + 1 + 1] > cs) {
              cs = F[(t + 1)][0 + 1][w + 1][p + 1 + 1];
              xs = 0;
            }
          }

          // [X_t=1] Consider running the job at time t
          // Ignore p_min at startup
          if ((z == 0 && w == 0 && p < p_max && w_min + 1 <= t && t <= w_max &&
               allow1[t] == 1) ||
              (z == 0 && w != 0 && w <= y_max - 1 && p_min <= p && p <= p_max &&
               w_min + 1 <= t && t <= w_max && allow1[t] == 1)) {
            if (c[t] + F[(t + 1)][1 + 1][w + 1 + 1][1 + 1] >= cs) {
              cs = c[t] + F[(t + 1)][1 + 1][w + 1 + 1][1 + 1];
              xs = 1;
            }
          }

          if (1 <= z && z <= t_max - 1 && w <= y_max && p < p_max &&
              t <= w_max && allow1[t] == 1) {
            if (c[t] + F[(t + 1)][z + 1 + 1][w + 1][p + 1 + 1] >= cs) {
              cs = c[t] + F[(t + 1)][z + 1 + 1][w + 1][p + 1 + 1];
              xs = 1;
            }
          }
          F[t][z + 1][w + 1][p + 1] = cs;
          X[t][z + 1][w + 1][p + 1] = xs;
        }
      }
    }
  }

  std::vector<std::vector<int>> Xopt(K + 1, std::vector<int>(T + 1, -1));

  int zx = 0;
  int px = 0;
  int wx = 0;
  int k = 0;
  for (int t = 1; t <= T; t++) {

    int xs = X[t][zx + 1][wx + 1][px + 1];

    Xopt[k][t] = xs;
    if (xs == 0) {
      if (zx == 0) {
        px = px + 1;
      } else {
        zx = 0;
        px = px + 1;
      }
    }
    if (xs == 1) {
      if (zx == 0) {
        zx = 1;
        wx = wx + 1;
        px = 1;
      } else {
        zx = zx + 1;
        px = px + 1;
      }
    }
  }
  k += 1;
  // #pragma omp parallel for
  for (int bound_actvation = y_min; bound_actvation <= y_max;
       bound_actvation++) {
    if (k == K) {
      break;
    }

    for (int bound_cpu = t_min; bound_cpu <= t_max; bound_cpu++) {
      if (k == K) {
        break;
      }

      // p_max proposital, pouco importante
      for (int bound_p = p_max; bound_p <= p_max; bound_p++) {
        if (k == K) {
          break;
        }

        int r_time = 0; // Running time
        int prd = 0;    // Period
        int n_acts = 0; // Number of activations

        for (int t = 1; t <= T; t++) {

          int xs = X[t][r_time + 1][n_acts + 1][prd + 1];

          Xopt[k][t] = xs;
          if (xs == 0) {
            // bounds no periodo times
            if ((allow1[t] == 1) && (r_time == 0) && (Xopt[k][t - 1] == 0) &&
                (prd >= bound_p) && (n_acts < bound_actvation) && t <= w_max &&
                t >= w_min + 1) {

              // cout << "entrou aqui" << prd << ',' << bound_p << endl;
              Xopt[k][t] = 1;
              n_acts = n_acts + 1;
              prd = 1;
              r_time = 1;
              continue;
            }

            if (r_time == 0) {
              prd = prd + 1;
            } else {
              r_time = 0;
              prd = prd + 1;
            }
          }
          if (xs == 1) {

            // bounds no activation times
            if ((r_time == 0) && (n_acts == bound_actvation + 1) &&
                (allow0[t] == 1)) {
              Xopt[k][t] = 0;
              prd = prd + 1;
              n_acts = n_acts;
              r_time = 0;
              continue;
            }

            // bounds on cpu running
            if ((r_time == bound_cpu) && (allow0[t] == 1)) {
              Xopt[k][t] = 0;
              r_time = 0;
              prd = prd + 1;
              continue;
            }

            if (r_time == 0) {
              r_time = 1;
              n_acts = n_acts + 1;
              prd = 1;
            } else {
              r_time = r_time + 1;
              prd = prd + 1;
            }
          }
        }
        k++;
      }
    }
  }

  double *final_column = new double[K * (T + 1)];

  std::vector<double> s(K + 1, 0);

  // #pragma omp parallel for
  for (int k = 0; k < K; k++) {
    for (int t = 1; t <= T; t++) {
      s[k] += c[t] * Xopt[k][t];
    }
  }

  // final_obj = s + c0;
  int h = 0;
  for (int k = 0; k < K; k++) {
    for (int t = 1; t <= T; t++) {
      final_column[h] = Xopt[k][t];
      h++;
    }
    final_column[h] = s[k] + c0;
    // cout << final_column[h];
    //         cout << endl;
    h++;
  }
  // delete[] final_column;
  return final_column;
}

/**
 * @brief Executes the parallelism algorithm to calculate the profile_total array.
 * 
 * This function launches multiple worker threads using std::async to calculate the profile_total array in parallel.
 * Each worker thread calculates a portion of the array based on the given parameters.
 * The results from the worker threads are stored in a vector and then copied to the profile_total array.
 * 
 * @param JOBS The number of jobs.
 * @param COVERS The number of covers.
 * @param UPPER_K The upper limit for K.
 * @param shadow_price An array of shadow prices.
 * @param allow An array of allow values.
 * @param qtde_cortes The number of cuts.
 * @param T The value of T.
 * @param priority An array of priority values.
 * @param uso_p An array of usage percentages.
 * @param min_statup An array of minimum startup values.
 * @param max_statup An array of maximum startup values.
 * @param min_cpu_time An array of minimum CPU time values.
 * @param max_cpu_time An array of maximum CPU time values.
 * @param min_periodo_job An array of minimum job period values.
 * @param max_periodo_job An array of maximum job period values.
 * @param win_min An array of minimum window values.
 * @param win_max An array of maximum window values.
 * @return double* A pointer to the profile_total array.
 */
extern "C" double *
paralelismo(const int JOBS, const int COVERS, const int UPPER_K,
            double shadow_price[], double allow[], int qtde_cortes, const int T,
            int priority[], double uso_p[], int min_statup[], int max_statup[],
            int min_cpu_time[], int max_cpu_time[], int min_periodo_job[],
            int max_periodo_job[], int win_min[], int win_max[]) {
  // Declare a vector to hold the worker threads

  std::vector<std::future<double *>> results;

  for (int job = 0; job < JOBS; job++) {

    // Declare a vector to hold the future results from the worker threads
    // Launch the worker threads using std::async
    //{
    results.push_back(std::async(
        pricing_dp, job, JOBS, COVERS, UPPER_K, shadow_price, allow,
        qtde_cortes, T, priority, uso_p, min_statup, max_statup, min_cpu_time,
        max_cpu_time, min_periodo_job, max_periodo_job, win_min, win_max));
    //}
  }

  int K = 0;
  for (int job = 0; job < JOBS; job++) {
    int K_ = max(max_statup[job] - min_statup[job], 1) *
             max(max_cpu_time[job] - min_cpu_time[job], 1) *
             max(max_periodo_job[job] - min_periodo_job[job], 1);
    if (K_ > UPPER_K) {
      K_ = UPPER_K;
    }
    K += K_;
  }
  // Get the results from the worker threads and store them in a vector
  // Allocate memory for the profile_total array
  double *profile_total = new double[K * 1 * (T + 1)];

  int h = 0;
  for (int job = 0; job < JOBS; job++) {
    // Get the results from the worker threads and store them in the
    // profile_total array

    auto profile_temp = results[job].get();
    int K_ = max(max_statup[job] - min_statup[job], 1) *
             max(max_cpu_time[job] - min_cpu_time[job], 1) *
             max(max_periodo_job[job] - min_periodo_job[job], 1);
    if (K_ > UPPER_K) {
      K_ = UPPER_K;
    }
    // #pragma omp parallel for
    for (int j = 0; j < K_ * (T + 1); j++) {
      profile_total[h] = profile_temp[j];
      h++;
    }
  }
  // delete[] profile_total;

  // Return the final result
  return profile_total;
}

/**
 * @brief Calculates the profile total for a given set of parameters.
 *
 * This function calculates the profile total by invoking the `pricing_dp`
 * function in a separate thread. It then retrieves the results from the worker
 * thread and stores them in the `profile_total` array.
 *
 * @param JOB The job parameter.
 * @param JOBS The number of jobs parameter.
 * @param COVERS The number of covers parameter.
 * @param UPPER_K The upper K parameter.
 * @param shadow_price An array of shadow prices.
 * @param allow An array of allow values.
 * @param qtde_cortes The number of cuts parameter.
 * @param T The T parameter.
 * @param priority An array of priority values.
 * @param uso_p An array of usage percentages.
 * @param min_statup An array of minimum startup values.
 * @param max_statup An array of maximum startup values.
 * @param min_cpu_time An array of minimum CPU time values.
 * @param max_cpu_time An array of maximum CPU time values.
 * @param min_periodo_job An array of minimum job period values.
 * @param max_periodo_job An array of maximum job period values.
 * @param win_min An array of minimum window values.
 * @param win_max An array of maximum window values.
 *
 * @return A pointer to the profile_total array.
 */
extern "C" double *single(const int JOB, const int JOBS, const int COVERS,
                          const int UPPER_K, double shadow_price[],
                          double allow[], int qtde_cortes, const int T,
                          int priority[], double uso_p[], int min_statup[],
                          int max_statup[], int min_cpu_time[],
                          int max_cpu_time[], int min_periodo_job[],
                          int max_periodo_job[], int win_min[], int win_max[]) {
  // Declare a vector to hold the worker threads

  std::vector<std::future<double *>> results;

  results.push_back(std::async(
      pricing_dp, JOB, JOBS, COVERS, UPPER_K, shadow_price, allow, qtde_cortes,
      T, priority, uso_p, min_statup, max_statup, min_cpu_time, max_cpu_time,
      min_periodo_job, max_periodo_job, win_min, win_max));

  int K = 1;

  // Get the results from the worker threads and store them in a vector
  // Allocate memory for the profile_total array
  double *profile_total = new double[K * 1 * (T + 1)];

  int h = 0;
  auto profile_temp = results[0].get();
  // #pragma omp parallel for
  for (int j = 0; j < K * (T + 1); j++) {
    profile_total[h] = profile_temp[j];
    h++;
  }

  // Return the final result
  return profile_total;
}

int main(int argc, const char *argv[]) { return 0; }
