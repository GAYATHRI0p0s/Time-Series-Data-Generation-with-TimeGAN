# Time-Series-Data-Generation-with-TimeGAN


### Household Electric Power Consumption (UCI / Kaggle) — MSc Data Analytics, Generative AI Capstone

---

## 1. Project Overview

Sequential data—from industrial sensor streams and financial prices to smart-meter readings—drives critical tasks like forecasting, anomaly detection, and operational planning. Yet, gaining access to these datasets is often difficult due to high collection costs or strict privacy concerns. Household electricity logs, for instance, can inadvertently expose personal details such as occupancy patterns and daily routines. Generative models capable of synthesizing realistic time-series sequences offer a practical solution, allowing researchers to develop, benchmark, and share models without exposing sensitive real-world records.

Early recurrent GANs designed for time series, such as C-RNN-GAN (Mogren, 2016) and RCGAN (Esteban et al., 2017), depended strictly on adversarial training, leaving them without an explicit mechanism to preserve step-by-step temporal transitions. TimeGAN (Yoon et al., 2019) overcomes this limitation by pairing an embedding network with a supervised next-step loss directly in latent space, proving effective across domains ranging from finance and healthcare to wind-power forecasting.

---

## 2. Dataset & Methodology

Time-series datasets, like household energy readings, are frequently scarce, privacy-sensitive, and hard to share openly. Generating synthetic sequences that faithfully preserve both the statistical properties and temporal patterns of real recordings offers a practical way forward. 

In this study, we apply TimeGAN (Yoon et al., 2019)—a generative framework that pairs adversarial training with a stepwise supervised loss—to the UCI Individual Household Electric Power Consumption dataset. We start with per-minute measurements across seven variables spanning roughly 47 months, which are cleaned, resampled into hourly averages, split chronologically, and segmented into 24-hour windows. Using a TensorFlow 2 implementation of the official TimeGAN architecture trained for 5,000 iterations per phase, we generate 10,000 synthetic daily sequences.

### Preprocessing Pipeline
1. **Timestamping:** The separate date and time columns are combined into a single chronological datetime index, sorted, and cleaned of any duplicate entries.
2. **Hourly resampling:** Per-minute readings are aggregated into hourly averages, reducing the series to 34,589 hours so that each 24-step window corresponds directly to a single day.
3. **Gap filling:** Any remaining missing hourly intervals are imputed using time-based linear interpolation.
4. **Scaling:** A Min–Max scaler is fitted exclusively on the training partition to map all features into $[0, 1]$—matching the range of TimeGAN's sigmoid output layers—while out-of-range test values are clipped to $[0, 1]$.
5. **Windowing:** Sliding overlapping windows of length $T = 24$ with a one-hour stride produces a tensor of shape $(N, 24, 7)$, from which a random subset of $10{,}000$ windows is sampled for model training and $6{,}895$ test windows are retained for evaluation.

---

## 3. Exploratory Data Analysis & Patterns

Figure 1 illustrates the key temporal patterns a generative model must capture: a pronounced diurnal cycle—characterized by low overnight consumption, a morning ramp-up starting around 06:00, and an evening peak near 20:00—alongside clear annual seasonality, with higher demand in winter, lower usage in summer, and a noticeable drop in August. Meanwhile, Figure 2 demonstrates that active power and current intensity are nearly perfectly correlated, while the individual sub-metering channels maintain strong positive relationships with overall power consumption.

---

## 4. Evaluation Framework & Results

We evaluate the generated data across multiple axes:
* **Fidelity:** Visual inspections, PCA and t-SNE latent manifold overlaps, correlation gap checks, and temporal autocorrelation (ACF) evaluations.
* **Utility:** A Train-on-Synthetic, Test-on-Real (TSTR) predictive evaluation compared against a Train-on-Real, Test-on-Real (TRTR) baseline.

We report these results transparently as observed, examine the underlying causes of the fidelity gap, and propose concrete steps for future improvement.

---

## 5. Objectives & Contributions

**Objectives:**
1. Apply an existing TimeGAN implementation to one real-world dataset.
2. Generate realistic synthetic time-series sequences.
3. Evaluate their resemblance to real data using standard metrics (visual comparison, PCA, t-SNE, discriminative and predictive scores).
4. Test whether synthetic data can substitute for real data in a downstream forecasting task.

**Contributions:**
* **End-to-End Reproducibility:** A complete, transparent pipeline transforming raw per-minute telemetry into fully evaluated synthetic sequences.
* **Multi-Dimensional Evaluation:** A thorough assessment spanning distributional coverage, inter-feature correlations, temporal structure, statistical fidelity, and downstream predictive utility.
* **Candid Diagnostic Insights:** A balanced evaluation highlighting both where TimeGAN excels on this dataset (marginal distributions, feature correlations, and manifold coverage) and where limitations persist (classifier indistinguishability and short-lag temporal dynamics).
