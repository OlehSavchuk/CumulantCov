#include "CumulantCov.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace stat_cov;

    std::vector<double> bins   = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    std::vector<double> counts = {3, 14, 48, 140, 310, 520, 690, 660, 490, 270, 115, 39, 10, 2, 1};

    HistogramCovEstimator<6> est(bins, counts);

    auto res_multi = est.evaluate_multinomial(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);
    auto res_delta = est.evaluate_standard_delta(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);

    const char* names[] = {"<N>", "omega", "S*sigma", "kappa*sigma^2"};
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== 1. Equivalence Check: Multinomial P_ij vs Standard Delta (J V J^T) ===\n";
    for (std::size_t i = 0; i < 4; ++i) {
        std::cout << std::setw(14) << names[i]
                  << " | Val: " << res_multi.value(i)
                  << " | SE(Multi): " << res_multi.error(i)
                  << " | SE(Delta): " << res_delta.error(i)
                  << " | Diff: " << std::abs(res_multi.error(i) - res_delta.error(i)) << "\n";
    }

    FactorialEfficiencyEstimator<4> eff_est(bins, counts, 0.75);
    auto res_eff = eff_est.evaluate(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);

    std::cout << std::setprecision(6) << "\n=== 2. Binomial Efficiency Unfolded (eps = 0.75) ===\n";
    for (std::size_t i = 0; i < 4; ++i) {
        std::cout << std::setw(14) << names[i] << " = "
                  << std::setw(10) << res_eff.value(i) << " +/- " << res_eff.error(i) << "\n";
    }

    std::vector<std::pair<double, double>> bins_2d = {
        {10, 4}, {10, 5}, {11, 5}, {11, 6}, {12, 5}, {12, 6}, {12, 7}, {13, 6}, {13, 7}
    };
    std::vector<double> counts_2d = {40, 95, 180, 210, 160, 320, 140, 190, 85};

    Joint2DCovEstimator joint_est(bins_2d, counts_2d);
    auto res_2d = joint_est.evaluate(
        obs::joint2d::C20,
        obs::joint2d::C11,
        obs::joint2d::C02,
        obs::joint2d::Chi11OverB2,
        obs::joint2d::C22
    );

    const char* names_2d[] = {"C20", "C11", "C02", "C11/C02", "C22"};
    std::cout << "\n=== 3. 2D Cross-Cumulants & Covariance ===\n";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << std::setw(14) << names_2d[i] << " = "
                  << std::setw(10) << res_2d.value(i) << " +/- " << res_2d.error(i) << "\n";
    }
    return 0;
}
