#include "CumulantCov.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace stat_cov;

void test_multinomial_vs_standard_delta() {
    std::vector<double> bins   = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<double> counts = {2, 9, 28, 75, 170, 310, 490, 620, 610, 470, 290, 140, 52, 16, 4, 1};

    HistogramCovEstimator<6> est(bins, counts);

    auto multi = est.evaluate_multinomial(
        obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2, obs::Skewness, obs::ExcessKurtosis
    );
    auto delta = est.evaluate_standard_delta(
        obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2, obs::Skewness, obs::ExcessKurtosis
    );

    for (std::size_t i = 0; i < 6; ++i) {
        assert(std::abs(multi.value(i) - delta.value(i)) < 1e-14);
        assert(std::abs(multi.error(i) - delta.error(i)) < 1e-14);
        for (std::size_t j = 0; j < 6; ++j) {
            assert(std::abs(multi.covariance(i, j) - delta.covariance(i, j)) < 1e-14);
            assert(std::abs(multi.correlation(i, j) - delta.correlation(i, j)) < 1e-13);
        }
    }
    std::cout << "[PASS] Multinomial vs Standard Delta (J V J^T) exact equivalence verified.\n";
}

void test_efficiency_unfolding_limit() {
    std::vector<double> bins   = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    std::vector<double> counts = {10, 35, 90, 180, 260, 240, 150, 70, 25, 6, 1};

    HistogramCovEstimator<4> direct_est(bins, counts);
    FactorialEfficiencyEstimator<4> eff_est_unit(bins, counts, 1.0); // eps = 1.0 must match direct

    auto res_dir = direct_est.evaluate(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);
    auto res_eff = eff_est_unit.evaluate(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);

    for (std::size_t i = 0; i < 4; ++i) {
        assert(std::abs(res_dir.value(i) - res_eff.value(i)) < 1e-12);
        assert(std::abs(res_dir.error(i) - res_eff.error(i)) < 1e-12);
        for (std::size_t j = 0; j < 4; ++j) {
            assert(std::abs(res_dir.covariance(i, j) - res_eff.covariance(i, j)) < 1e-12);
        }
    }

    // Check that eps < 1 increases statistical uncertainty as expected
    FactorialEfficiencyEstimator<4> eff_est_sub(bins, counts, 0.70);
    auto res_sub = eff_est_sub.evaluate(obs::Mean, obs::Omega, obs::SSigma, obs::KappaSigma2);
    assert(res_sub.error(3) > res_eff.error(3));
    std::cout << "[PASS] FactorialEfficiencyEstimator unit limit & unfolding variance verified.\n";
}

void test_joint_2d_cross_cumulants() {
    std::vector<std::pair<double, double>> bins_2d = {
        {8, 3}, {9, 4}, {10, 4}, {10, 5}, {11, 5}, {11, 6}, {12, 6}, {13, 7}
    };
    std::vector<double> counts_2d = {25, 80, 150, 220, 260, 190, 110, 40};

    Joint2DCovEstimator joint(bins_2d, counts_2d);
    auto res = joint.evaluate(
        obs::joint2d::C20,
        obs::joint2d::C11,
        obs::joint2d::C02,
        obs::joint2d::Chi11OverB2,
        obs::joint2d::C22
    );

    for (std::size_t i = 0; i < 5; ++i) {
        assert(res.error(i) > 0.0);
        assert(std::abs(res.correlation(i, i) - 1.0) < 1e-14);
        for (std::size_t j = 0; j < 5; ++j) {
            assert(std::abs(res.covariance(i, j) - res.covariance(j, i)) < 1e-15);
            assert(std::abs(res.correlation(i, j)) <= 1.0 + 1e-12);
        }
    }
    std::cout << "[PASS] Joint2DCovEstimator symmetry and Cauchy-Schwarz bounds verified.\n";
}

int main() {
    test_multinomial_vs_standard_delta();
    test_efficiency_unfolding_limit();
    test_joint_2d_cross_cumulants();
    std::cout << "All CumulantCov unit tests passed!\n";
    return 0;
}