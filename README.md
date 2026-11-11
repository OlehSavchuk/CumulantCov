# CumulantCov

[![CI](https://github.com/OlehSavchuk/CumulantCov/actions/workflows/ci.yml/badge.svg)](https://github.com/OlehSavchuk/CumulantCov/actions)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**`CumulantCov`** is a zero-dependency, single-header C++17 library for computing **asymptotic standard errors, full covariance matrices, and Pearson correlation matrices** of arbitrary moments, cumulants, intensive cumulant ratios ($\omega, S\sigma, \kappa\sigma^2, \kappa_6/\kappa_2$), and joint cross-cumulants ($C_{pq}$).

Instead of relying on hardcoded, error-prone 8th- to 12th-order central moment algebraic tables, `CumulantCov` pairs **compile-time forward-mode automatic differentiation (dual numbers)** with the exact **multinomial histogram covariance identity**:

$$\text{Cov}(\hat{P}_i, \hat{P}_j) = \frac{1}{N_{\text{ev}}}\left(P_i \delta_{ij} - P_i P_j\right)$$

It also ships with a reference implementation of the classical **Stuart & Ord / Luo central-moment Delta theorem** ($\mathbf{J} \mathbf{V} \mathbf{J}^T$) to verify exact machine-precision equivalence ($< 10^{-15}$ relative difference).

---

## Key Features

- **Full $M \times M$ Covariance & Correlation Matrices:** Simultaneously computes values, standard errors, and off-diagonal error covariances $\text{Cov}(\hat{\mathcal{O}}_a, \hat{\mathcal{O}}_b)$ across any number of observables in a single pass.
- **Dual Formulations Built-In:**
  1. `evaluate_multinomial(...)`: $O(B)$ contraction over non-empty histogram bins requiring central moments only up to order $K$.
  2. `evaluate_standard_delta(...)`: Classical $\mathbf{J}\mathbf{V}\mathbf{J}^T$ contraction using central moments up to order $2K$.
- **Arbitrary Custom Observables via Auto-Diff:** Pass any C++ lambda `f(m)` written in terms of the central moment array `m`; exact analytical partial derivatives $\partial \mathcal{O}_a / \partial m_r$ are propagated automatically at compile time.
- **Analytical Binomial Efficiency Unfolding (`FactorialEfficiencyEstimator`):** Unfolds binomial detector tracking efficiency $\epsilon \in (0, 1]$ at the factorial moment level ($F_r = f_r / \epsilon^r$), transforms to true central moments via Stirling numbers of the second kind, and propagates the exact unfolded covariance matrix.
- **Sparse 2D Joint Distributions (`Joint2DCovEstimator`):** Computes off-diagonal cross-cumulants ($C_{11}, C_{21}, C_{22}$) and susceptibility ratios (e.g., $\chi_{11}^{AB}/\chi_2^B$) with full error covariance.

---

## Quick Start

### 1. Include the Header
Copy `include/CumulantCov.hpp` into your project, or include it via CMake `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    CumulantCov
    GIT_repository [https://github.com/OlehSavchuk/CumulantCov.git](https://github.com/OlehSavchuk/CumulantCov.git)
    GIT_TAG main
)
FetchContent_MakeAvailable(CumulantCov)
target_link_libraries(your_target PRIVATE CumulantCov::CumulantCov)
```

### 2. Compute Cumulant Ratios & Full Covariance Matrix

```cpp
#include "CumulantCov.hpp"
#include <iostream>

int main() {
    using namespace stat_cov;

    std::vector<double> bins   = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<double> counts = {15, 52, 140, 310, 520, 690, 660, 480, 250, 95, 22};

    HistogramCovEstimator<6> est(bins, counts);

    // Evaluate standard cumulant ratios + any custom lambda observable
    auto res = est.evaluate(
        obs::Mean,
        obs::Omega,        // kappa_2 / kappa_1
        obs::SSigma,       // kappa_3 / kappa_2
        obs::KappaSigma2,  // kappa_4 / kappa_2
        [](const auto& m) { return obs::Kappa6(m) / obs::Kappa2(m); } // kappa_6 / kappa_2
    );

    std::cout << "S*sigma       = " << res.value(2) << " +/- " << res.error(2) << "\n";
    std::cout << "kappa*sigma^2 = " << res.value(3) << " +/- " << res.error(3) << "\n";
    std::cout << "Cov(S*sigma, kappa*sigma^2)  = " << res.covariance(2, 3) << "\n";
    std::cout << "Corr(S*sigma, kappa*sigma^2) = " << res.correlation(2, 3) << "\n";
}
```

---

## Building Examples and Running Unit Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/cumulant_demo
```

---

## Mathematical Formulation

Let $\hat{P}_i = n_i / N_{\text{ev}}$ be the empirical probability in multiplicity bin $N_i$. The central moment of order $r$ is $m_r(\hat{\mathbf{P}}) = \sum_i (N_i - \bar{N})^r \hat{P}_i$. Its bin-level influence function (Fréchet derivative) is:

$$d_{r,i} = \frac{\partial m_r}{\partial \hat{P}_i} = \Delta_i^r - r \mu_{r-1} \Delta_i - \mu_r, \qquad \Delta_i = N_i - \bar{N}$$

For any collection of observables $\mathcal{O}_a(\mathbf{m})$, dual-number evaluation yields the exact Jacobian $J_{a,r} = \partial \mathcal{O}_a / \partial m_r$. Projecting onto the histogram bins gives $g_{i,a} = \sum_{r=1}^K J_{a,r} d_{r,i}$, and contracting with the multinomial covariance $\text{Cov}(\hat{P}_i, \hat{P}_j) = \frac{1}{N_{\text{ev}}}(P_i \delta_{ij} - P_i P_j)$ reduces to an $O(B)$ expectation value:

$$\text{Cov}(\hat{\mathcal{O}}_a, \hat{\mathcal{O}}_b) = \frac{1}{N_{\text{ev}}} \left[ \sum_{i=1}^B P_i g_{i,a} g_{i,b} - \left(\sum_{i=1}^B P_i g_{i,a}\right)\left(\sum_{j=1}^B P_j g_{j,b}\right) \right]$$

---

## Author & License

Developed by **Oleh Savchuk** (Facility for Rare Isotope Beams, Michigan State University).  
Released under the [MIT License](LICENSE).