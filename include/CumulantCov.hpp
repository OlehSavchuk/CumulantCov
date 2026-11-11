#ifndef CUMULANT_COV_HPP
#define CUMULANT_COV_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stat_cov {

// ============================================================================
// 1. Compile-Time Forward-Mode Auto-Diff (Dual Numbers)
//    Computes exact analytical partial derivatives dO_a / dm_r.
// ============================================================================
template <std::size_t K>
struct Dual {
    double val = 0.0;
    std::array<double, K> grad{};

    constexpr Dual() = default;
    constexpr explicit Dual(double v) : val(v) { grad.fill(0.0); }
    constexpr Dual(double v, std::size_t idx) : val(v) {
        grad.fill(0.0);
        if (idx < K) grad[idx] = 1.0;
    }

    friend constexpr Dual operator+(const Dual& a, const Dual& b) {
        Dual res(a.val + b.val);
        for (std::size_t i = 0; i < K; ++i) res.grad[i] = a.grad[i] + b.grad[i];
        return res;
    }
    friend constexpr Dual operator+(const Dual& a, double b) { return a + Dual(b); }
    friend constexpr Dual operator+(double a, const Dual& b) { return Dual(a) + b; }

    friend constexpr Dual operator-(const Dual& a, const Dual& b) {
        Dual res(a.val - b.val);
        for (std::size_t i = 0; i < K; ++i) res.grad[i] = a.grad[i] - b.grad[i];
        return res;
    }
    friend constexpr Dual operator-(const Dual& a, double b) { return a - Dual(b); }
    friend constexpr Dual operator-(double a, const Dual& b) { return Dual(a) - b; }
    friend constexpr Dual operator-(const Dual& a) { return Dual(0.0) - a; }

    friend constexpr Dual operator*(const Dual& a, const Dual& b) {
        Dual res(a.val * b.val);
        for (std::size_t i = 0; i < K; ++i)
            res.grad[i] = a.grad[i] * b.val + a.val * b.grad[i];
        return res;
    }
    friend constexpr Dual operator*(const Dual& a, double b) { return a * Dual(b); }
    friend constexpr Dual operator*(double a, const Dual& b) { return Dual(a) * b; }

    friend constexpr Dual operator/(const Dual& a, const Dual& b) {
        Dual res(a.val / b.val);
        const double inv_b2 = 1.0 / (b.val * b.val);
        for (std::size_t i = 0; i < K; ++i)
            res.grad[i] = (a.grad[i] * b.val - a.val * b.grad[i]) * inv_b2;
        return res;
    }
    friend constexpr Dual operator/(const Dual& a, double b) { return a / Dual(b); }
    friend constexpr Dual operator/(double a, const Dual& b) { return Dual(a) / b; }

    friend Dual pow(const Dual& a, double p) {
        double pv = std::pow(a.val, p);
        double dp = p * std::pow(a.val, p - 1.0);
        Dual res(pv);
        for (std::size_t i = 0; i < K; ++i) res.grad[i] = dp * a.grad[i];
        return res;
    }

    friend Dual sqrt(const Dual& a) { return pow(a, 0.5); }
};

// ============================================================================
// 2. Result Container: Values, Standard Errors, Covariance & Correlation
// ============================================================================
template <std::size_t M>
struct CovResult {
    std::array<double, M> values{};
    std::array<double, M> errors{};
    std::array<std::array<double, M>, M> cov{};
    std::array<std::array<double, M>, M> corr{};

    double value(std::size_t i) const { return values.at(i); }
    double error(std::size_t i) const { return errors.at(i); }
    double covariance(std::size_t i, std::size_t j) const { return cov.at(i).at(j); }
    double correlation(std::size_t i, std::size_t j) const { return corr.at(i).at(j); }

    void finalize_errors_and_correlations() {
        for (std::size_t a = 0; a < M; ++a)
            errors[a] = std::sqrt(std::max(0.0, cov[a][a]));
        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t b = 0; b < M; ++b) {
                double denom = errors[a] * errors[b];
                corr[a][b] = (denom > 0.0) ? (cov[a][b] / denom) : 0.0;
            }
        }
    }
};

// ============================================================================
// 3. 1D Estimator: Both Multinomial (P_i) & Standard Delta Theorem (J V J^T)
//    State vector m = { mu'_1 (mean), mu_2, mu_3, ..., mu_K }
// ============================================================================
template <std::size_t K = 6>
class HistogramCovEstimator {
    static_assert(K >= 2, "Max moment order K must be at least 2.");

private:
    std::vector<double> x_bins_;
    std::vector<double> p_bins_;
    double n_ev_ = 0.0;

    std::array<double, K> moments_{};                 // m[0] = mean, m[r-1] = mu_r (r = 2..K)
    std::array<double, 2 * K + 1> central_mu_2K_{};   // Central moments mu_0 .. mu_{2K} for standard Delta
    std::array<std::array<double, K>, K> V_moments_{};// Exact Stuart & Ord / Luo moment covariance matrix
    std::vector<std::array<double, K>> d_ir_;         // Bin influence functions: d m_r / d P_i

    void precompute_all() {
        const std::size_t B = x_bins_.size();
        moments_.fill(0.0);
        central_mu_2K_.fill(0.0);

        // 1. Sample Mean (mu'_1)
        double mean = 0.0;
        for (std::size_t i = 0; i < B; ++i) mean += x_bins_[i] * p_bins_[i];
        moments_[0] = mean;

        // 2. Central Moments mu_r up to 2K (for Standard Delta Theorem)
        std::vector<double> delta(B);
        for (std::size_t i = 0; i < B; ++i) delta[i] = x_bins_[i] - mean;

        central_mu_2K_[0] = 1.0; // mu_0 = 1
        central_mu_2K_[1] = 0.0; // mu_1 = 0
        for (std::size_t r = 2; r <= 2 * K; ++r) {
            double mu_r = 0.0;
            for (std::size_t i = 0; i < B; ++i)
                mu_r += p_bins_[i] * std::pow(delta[i], static_cast<int>(r));
            central_mu_2K_[r] = mu_r;
            if (r <= K) moments_[r - 1] = mu_r;
        }

        // 3. Explicit Stuart & Ord / Luo Moment Covariance Matrix V_{a,b}
        const double mu2 = central_mu_2K_[2];
        V_moments_[0][0] = mu2 / n_ev_;

        for (std::size_t r = 2; r <= K; ++r) {
            const double rd = static_cast<double>(r);
            const double cov_mean_mr = (central_mu_2K_[r + 1] - rd * central_mu_2K_[r - 1] * mu2) / n_ev_;
            V_moments_[0][r - 1] = cov_mean_mr;
            V_moments_[r - 1][0] = cov_mean_mr;

            for (std::size_t s = r; s <= K; ++s) {
                const double sd = static_cast<double>(s);
                const double term = central_mu_2K_[r + s]
                                  - central_mu_2K_[r] * central_mu_2K_[s]
                                  - rd * central_mu_2K_[r - 1] * central_mu_2K_[s + 1]
                                  - sd * central_mu_2K_[r + 1] * central_mu_2K_[s - 1]
                                  + rd * sd * central_mu_2K_[r - 1] * central_mu_2K_[s - 1] * mu2;
                V_moments_[r - 1][s - 1] = term / n_ev_;
                V_moments_[s - 1][r - 1] = term / n_ev_;
            }
        }

        // 4. Bin-level influence functions d_{r,i} = d m_r / d P_i (for Multinomial Engine)
        d_ir_.assign(B, std::array<double, K>{});
        for (std::size_t i = 0; i < B; ++i) {
            const double d = delta[i];
            d_ir_[i][0] = d;
            double d_pow = d;
            for (std::size_t r = 2; r <= K; ++r) {
                d_pow *= d;
                const double mu_r_minus_1 = central_mu_2K_[r - 1];
                d_ir_[i][r - 1] = d_pow - static_cast<double>(r) * mu_r_minus_1 * d - central_mu_2K_[r];
            }
        }
    }

public:
    HistogramCovEstimator() = default;

    HistogramCovEstimator(const std::vector<double>& bin_values,
                          const std::vector<double>& bin_counts) {
        set_histogram(bin_values, bin_counts);
    }

    explicit HistogramCovEstimator(const std::vector<double>& raw_events) {
        std::map<double, double> hist;
        for (double v : raw_events) hist[v] += 1.0;
        std::vector<double> vals, counts;
        vals.reserve(hist.size());
        counts.reserve(hist.size());
        for (const auto& [val, cnt] : hist) {
            vals.push_back(val);
            counts.push_back(cnt);
        }
        set_histogram(vals, counts);
    }

    void set_histogram(const std::vector<double>& bin_values,
                       const std::vector<double>& bin_counts) {
        if (bin_values.size() != bin_counts.size() || bin_values.empty())
            throw std::invalid_argument("Bin values and counts must be non-empty and match in size.");

        x_bins_ = bin_values;
        p_bins_.resize(bin_counts.size());
        n_ev_ = 0.0;
        for (double c : bin_counts) n_ev_ += c;
        if (n_ev_ <= 0.0) throw std::invalid_argument("Total number of events N_ev must be positive.");

        for (std::size_t i = 0; i < bin_counts.size(); ++i)
            p_bins_[i] = bin_counts[i] / n_ev_;

        precompute_all();
    }

    const std::array<std::array<double, K>, K>& moment_covariance_matrix() const {
        return V_moments_;
    }

    template <typename... Funcs>
    CovResult<sizeof...(Funcs)> evaluate_multinomial(Funcs... funcs) const {
        constexpr std::size_t M = sizeof...(Funcs);
        CovResult<M> res;

        std::array<Dual<K>, K> dual_m;
        for (std::size_t r = 0; r < K; ++r) dual_m[r] = Dual<K>(moments_[r], r);

        std::array<Dual<K>, M> dual_obs = { funcs(dual_m)... };
        for (std::size_t a = 0; a < M; ++a) res.values[a] = dual_obs[a].val;

        const std::size_t B = x_bins_.size();
        std::vector<std::array<double, M>> grad_P(B);
        std::array<double, M> mean_grad{};

        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t a = 0; a < M; ++a) {
                double g_ia = 0.0;
                for (std::size_t r = 0; r < K; ++r)
                    g_ia += dual_obs[a].grad[r] * d_ir_[i][r];
                grad_P[i][a] = g_ia;
                mean_grad[a] += p_bins_[i] * g_ia;
            }
        }

        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t b = a; b < M; ++b) {
                double cross = 0.0;
                for (std::size_t i = 0; i < B; ++i)
                    cross += p_bins_[i] * grad_P[i][a] * grad_P[i][b];

                double cov_ab = (cross - mean_grad[a] * mean_grad[b]) / n_ev_;
                res.cov[a][b] = cov_ab;
                res.cov[b][a] = cov_ab;
            }
        }
        res.finalize_errors_and_correlations();
        return res;
    }

    template <typename... Funcs>
    CovResult<sizeof...(Funcs)> evaluate_standard_delta(Funcs... funcs) const {
        constexpr std::size_t M = sizeof...(Funcs);
        CovResult<M> res;

        std::array<Dual<K>, K> dual_m;
        for (std::size_t r = 0; r < K; ++r) dual_m[r] = Dual<K>(moments_[r], r);

        std::array<Dual<K>, M> dual_obs = { funcs(dual_m)... };
        for (std::size_t a = 0; a < M; ++a) res.values[a] = dual_obs[a].val;

        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t b = a; b < M; ++b) {
                double cov_ab = 0.0;
                for (std::size_t r = 0; r < K; ++r) {
                    for (std::size_t s = 0; s < K; ++s) {
                        cov_ab += dual_obs[a].grad[r] * V_moments_[r][s] * dual_obs[b].grad[s];
                    }
                }
                res.cov[a][b] = cov_ab;
                res.cov[b][a] = cov_ab;
            }
        }
        res.finalize_errors_and_correlations();
        return res;
    }

    template <typename... Funcs>
    CovResult<sizeof...(Funcs)> evaluate(Funcs... funcs) const {
        return evaluate_multinomial(funcs...);
    }
};

// ============================================================================
// 4. Binomial Efficiency-Corrected Estimator (Factorial Cumulant Unfolding)
//    Unfolds detector efficiency eps in (0, 1] via F_r = f_r / eps^r,
//    converts F_r -> Factorial Cumulants fc_r -> Cumulants kappa_r -> Central Moments mu_r
//    to avoid large <N>^r catastrophic cancellation in raw-moment subtraction.
// ============================================================================
template <std::size_t K = 4>
class FactorialEfficiencyEstimator {
    static_assert(K >= 2 && K <= 6, "FactorialEfficiencyEstimator supports K in [2, 6].");

private:
    std::vector<double> n_bins_;
    std::vector<double> p_bins_;
    double n_ev_ = 0.0;
    double eps_  = 1.0;

    std::array<double, K> unfolded_F_{};          // True factorial moments F_1 .. F_K
    std::vector<std::array<double, K>> phi_ir_;   // True factorial polynomials per bin i: n_i^{[r]} / eps^r

    static constexpr double stirling2(std::size_t r, std::size_t j) {
        if (j == 0 || j > r) return 0.0;
        if (j == 1 || j == r) return 1.0;
        return static_cast<double>(j) * stirling2(r - 1, j) + stirling2(r - 1, j - 1);
    }

    static constexpr double binom_coeff(std::size_t n, std::size_t k) {
        if (k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        double res = 1.0;
        for (std::size_t i = 1; i <= k; ++i)
            res = res * static_cast<double>(n - i + 1) / static_cast<double>(i);
        return res;
    }

    static double falling_factorial(double n, std::size_t r) {
        double prod = 1.0;
        for (std::size_t k = 0; k < r; ++k) prod *= (n - static_cast<double>(k));
        return prod;
    }

    // Numerically stable transform:
    //   1. Unfolded Factorial Moments F_r -> Factorial Cumulants fc_r (cumulant recursion)
    //      fc_r = F_r - sum_{m=1}^{r-1} C(r-1, m-1) fc_m F_{r-m}
    //   2. Factorial Cumulants fc_r -> Ordinary Cumulants kappa_r
    //      kappa_r = sum_{j=1}^r S(r, j) fc_j
    //   3. Ordinary Cumulants kappa_r -> Central Moments mu_r (Bell polynomial recursion, independent of <N>!)
    //      mu_r = sum_{m=1}^{r-1} C(r-1, m-1) kappa_m mu_{r-m}  (with mu_0 = 1, mu_1 = 0)
    static std::array<Dual<K>, K> factorial_to_central_moments(const std::array<Dual<K>, K>& F) {
        // Step 1: F_r -> Factorial Cumulants fc_r (1-indexed in array of size K+1)
        std::array<Dual<K>, K + 1> fc{};
        for (std::size_t r = 1; r <= K; ++r) {
            Dual<K> val = F[r - 1];
            for (std::size_t m = 1; m < r; ++m) {
                val = val - binom_coeff(r - 1, m - 1) * fc[m] * F[r - m - 1];
            }
            fc[r] = val;
        }

        // Step 2: fc_r -> Ordinary Cumulants kappa_r
        std::array<Dual<K>, K + 1> kappa{};
        for (std::size_t r = 1; r <= K; ++r) {
            Dual<K> sum(0.0);
            for (std::size_t j = 1; j <= r; ++j) {
                sum = sum + stirling2(r, j) * fc[j];
            }
            kappa[r] = sum;
        }

        // Step 3: kappa_r -> Central Moments m = {<N>, mu_2, ..., mu_K}
        std::array<Dual<K>, K + 1> mu{};
        mu[0] = Dual<K>(1.0);
        mu[1] = Dual<K>(0.0); // Central moment mu_1 = 0
        for (std::size_t r = 2; r <= K; ++r) {
            Dual<K> sum(0.0);
            for (std::size_t m = 2; m <= r; ++m) { // m=1 term vanishes since mu[1] = 0
                sum = sum + binom_coeff(r - 1, m - 1) * kappa[m] * mu[r - m];
            }
            mu[r] = sum;
        }

        std::array<Dual<K>, K> m_out{};
        m_out[0] = kappa[1]; // Mean <N>
        for (std::size_t r = 2; r <= K; ++r) m_out[r - 1] = mu[r];
        return m_out;
    }

public:
    FactorialEfficiencyEstimator(const std::vector<double>& measured_bins,
                                 const std::vector<double>& measured_counts,
                                 double efficiency)
        : n_bins_(measured_bins), eps_(efficiency) {
        if (measured_bins.size() != measured_counts.size() || measured_bins.empty())
            throw std::invalid_argument("Measured bins and counts must match and be non-empty.");
        if (eps_ <= 0.0 || eps_ > 1.0)
            throw std::invalid_argument("Efficiency eps must be in (0, 1].");

        const std::size_t B = measured_counts.size();
        p_bins_.resize(B);
        n_ev_ = 0.0;
        for (double c : measured_counts) n_ev_ += c;
        for (std::size_t i = 0; i < B; ++i) p_bins_[i] = measured_counts[i] / n_ev_;

        unfolded_F_.fill(0.0);
        phi_ir_.assign(B, std::array<double, K>{});
        for (std::size_t r = 1; r <= K; ++r) {
            const double inv_eps_r = 1.0 / std::pow(eps_, static_cast<int>(r));
            for (std::size_t i = 0; i < B; ++i) {
                phi_ir_[i][r - 1] = falling_factorial(n_bins_[i], r) * inv_eps_r;
                unfolded_F_[r - 1] += p_bins_[i] * phi_ir_[i][r - 1];
            }
        }
    }

    template <typename... Funcs>
    CovResult<sizeof...(Funcs)> evaluate(Funcs... funcs) const {
        constexpr std::size_t M = sizeof...(Funcs);
        CovResult<M> res;

        std::array<Dual<K>, K> dual_F;
        for (std::size_t r = 0; r < K; ++r) dual_F[r] = Dual<K>(unfolded_F_[r], r);

        std::array<Dual<K>, K> dual_m = factorial_to_central_moments(dual_F);

        std::array<Dual<K>, M> dual_obs = { funcs(dual_m)... };
        for (std::size_t a = 0; a < M; ++a) res.values[a] = dual_obs[a].val;

        const std::size_t B = n_bins_.size();
        std::vector<std::array<double, M>> grad_P(B);
        std::array<double, M> mean_grad{};

        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t a = 0; a < M; ++a) {
                double g_ia = 0.0;
                for (std::size_t r = 0; r < K; ++r)
                    g_ia += dual_obs[a].grad[r] * phi_ir_[i][r];
                grad_P[i][a] = g_ia;
                mean_grad[a] += p_bins_[i] * g_ia;
            }
        }

        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t b = a; b < M; ++b) {
                double cross = 0.0;
                for (std::size_t i = 0; i < B; ++i)
                    cross += p_bins_[i] * grad_P[i][a] * grad_P[i][b];

                double cov_ab = (cross - mean_grad[a] * mean_grad[b]) / n_ev_;
                res.cov[a][b] = cov_ab;
                res.cov[b][a] = cov_ab;
            }
        }
        res.finalize_errors_and_correlations();
        return res;
    }
};

// ============================================================================
// 5. 2D Joint Distribution Estimator: Cross-Cumulants & Susceptibilities
// ============================================================================
class Joint2DCovEstimator {
public:
    static constexpr std::size_t L = 14;

private:
    std::vector<std::pair<double, double>> bins_ab_;
    std::vector<double> p_bins_;
    double n_ev_ = 0.0;

    std::array<double, L> moments_{};
    std::vector<std::array<double, L>> d_ir_;

    static constexpr int pq_index(int p, int q) {
        int order = p + q;
        if (order == 1) return (p == 1) ? 0 : 1;
        if (order == 2) return 2 + q;
        if (order == 3) return 5 + q;
        if (order == 4) return 9 + q;
        return -1;
    }

    double get_central_mu(int p, int q) const {
        if (p < 0 || q < 0) return 0.0;
        if (p == 0 && q == 0) return 1.0;
        if (p + q == 1) return 0.0;
        int idx = pq_index(p, q);
        return (idx >= 2) ? moments_[static_cast<std::size_t>(idx)] : 0.0;
    }

    void precompute_joint() {
        const std::size_t B = bins_ab_.size();
        moments_.fill(0.0);

        double mean_a = 0.0, mean_b = 0.0;
        for (std::size_t i = 0; i < B; ++i) {
            mean_a += p_bins_[i] * bins_ab_[i].first;
            mean_b += p_bins_[i] * bins_ab_[i].second;
        }
        moments_[0] = mean_a;
        moments_[1] = mean_b;

        std::vector<double> da(B), db(B);
        for (std::size_t i = 0; i < B; ++i) {
            da[i] = bins_ab_[i].first - mean_a;
            db[i] = bins_ab_[i].second - mean_b;
        }

        for (int order = 2; order <= 4; ++order) {
            for (int q = 0; q <= order; ++q) {
                int p = order - q;
                int idx = pq_index(p, q);
                double val = 0.0;
                for (std::size_t i = 0; i < B; ++i)
                    val += p_bins_[i] * std::pow(da[i], p) * std::pow(db[i], q);
                moments_[static_cast<std::size_t>(idx)] = val;
            }
        }

        d_ir_.assign(B, std::array<double, L>{});
        for (std::size_t i = 0; i < B; ++i) {
            d_ir_[i][0] = da[i];
            d_ir_[i][1] = db[i];
            for (int order = 2; order <= 4; ++order) {
                for (int q = 0; q <= order; ++q) {
                    int p = order - q;
                    int idx = pq_index(p, q);
                    double term = std::pow(da[i], p) * std::pow(db[i], q)
                                - static_cast<double>(p) * get_central_mu(p - 1, q) * da[i]
                                - static_cast<double>(q) * get_central_mu(p, q - 1) * db[i]
                                - moments_[static_cast<std::size_t>(idx)];
                    d_ir_[i][static_cast<std::size_t>(idx)] = term;
                }
            }
        }
    }

public:
    Joint2DCovEstimator(const std::vector<std::pair<double, double>>& bin_pairs,
                        const std::vector<double>& bin_counts)
        : bins_ab_(bin_pairs) {
        if (bin_pairs.size() != bin_counts.size() || bin_pairs.empty())
            throw std::invalid_argument("2D bin pairs and counts must match and be non-empty.");

        p_bins_.resize(bin_counts.size());
        n_ev_ = 0.0;
        for (double c : bin_counts) n_ev_ += c;
        for (std::size_t i = 0; i < bin_counts.size(); ++i)
            p_bins_[i] = bin_counts[i] / n_ev_;

        precompute_joint();
    }

    template <typename... Funcs>
    CovResult<sizeof...(Funcs)> evaluate(Funcs... funcs) const {
        constexpr std::size_t M = sizeof...(Funcs);
        CovResult<M> res;

        std::array<Dual<L>, L> dual_m;
        for (std::size_t r = 0; r < L; ++r) dual_m[r] = Dual<L>(moments_[r], r);

        std::array<Dual<L>, M> dual_obs = { funcs(dual_m)... };
        for (std::size_t a = 0; a < M; ++a) res.values[a] = dual_obs[a].val;

        const std::size_t B = bins_ab_.size();
        std::vector<std::array<double, M>> grad_P(B);
        std::array<double, M> mean_grad{};

        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t a = 0; a < M; ++a) {
                double g_ia = 0.0;
                for (std::size_t r = 0; r < L; ++r)
                    g_ia += dual_obs[a].grad[r] * d_ir_[i][r];
                grad_P[i][a] = g_ia;
                mean_grad[a] += p_bins_[i] * g_ia;
            }
        }

        for (std::size_t a = 0; a < M; ++a) {
            for (std::size_t b = a; b < M; ++b) {
                double cross = 0.0;
                for (std::size_t i = 0; i < B; ++i)
                    cross += p_bins_[i] * grad_P[i][a] * grad_P[i][b];

                double cov_ab = (cross - mean_grad[a] * mean_grad[b]) / n_ev_;
                res.cov[a][b] = cov_ab;
                res.cov[b][a] = cov_ab;
            }
        }
        res.finalize_errors_and_correlations();
        return res;
    }
};

// ============================================================================
// 6. Pre-Built 1D and 2D Physics & Statistics Observable Functors
// ============================================================================
namespace obs {
    inline constexpr auto Mean   = [](const auto& m) { return m[0]; };
    inline constexpr auto Kappa1 = [](const auto& m) { return m[0]; };
    inline constexpr auto Kappa2 = [](const auto& m) { return m[1]; };
    inline constexpr auto Kappa3 = [](const auto& m) { return m[2]; };
    inline constexpr auto Kappa4 = [](const auto& m) { return m[3] - 3.0 * m[1] * m[1]; };
    inline constexpr auto Kappa5 = [](const auto& m) { return m[4] - 10.0 * m[2] * m[1]; };
    inline constexpr auto Kappa6 = [](const auto& m) {
        return m[5] - 15.0 * m[3] * m[1] - 10.0 * m[2] * m[2] + 30.0 * m[1] * m[1] * m[1];
    };

    inline constexpr auto Omega       = [](const auto& m) { return Kappa2(m) / Kappa1(m); };
    inline constexpr auto SSigma      = [](const auto& m) { return Kappa3(m) / Kappa2(m); };
    inline constexpr auto KappaSigma2 = [](const auto& m) { return Kappa4(m) / Kappa2(m); };
    inline constexpr auto Skewness    = [](const auto& m) { return m[2] / pow(m[1], 1.5); };
    inline constexpr auto ExcessKurtosis = [](const auto& m) { return (m[3] / (m[1] * m[1])) - 3.0; };

    namespace joint2d {
        inline constexpr auto MeanA = [](const auto& m) { return m[0]; };
        inline constexpr auto MeanB = [](const auto& m) { return m[1]; };
        inline constexpr auto C20   = [](const auto& m) { return m[2]; };
        inline constexpr auto C11   = [](const auto& m) { return m[3]; };
        inline constexpr auto C02   = [](const auto& m) { return m[4]; };
        inline constexpr auto C21   = [](const auto& m) { return m[6]; };
        inline constexpr auto C12   = [](const auto& m) { return m[7]; };
        inline constexpr auto C40   = [](const auto& m) { return m[9] - 3.0 * m[2] * m[2]; };
        inline constexpr auto C31   = [](const auto& m) { return m[10] - 3.0 * m[2] * m[3]; };
        inline constexpr auto C22   = [](const auto& m) { return m[11] - m[2] * m[4] - 2.0 * m[3] * m[3]; };

        inline constexpr auto Chi11OverB2 = [](const auto& m) { return C11(m) / C02(m); };
        inline constexpr auto Chi22OverB2 = [](const auto& m) { return C22(m) / C02(m); };
    } // namespace joint2d
} // namespace obs

} // namespace stat_cov

#endif // CUMULANT_COV_HPP
