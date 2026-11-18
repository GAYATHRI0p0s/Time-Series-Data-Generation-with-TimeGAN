# Time Series Data Generation with TimeGAN

### Generative AI Capstone Project — MSc in Data Analytics

---

## 1. Project Overview & Course Details

* **Project Title:** Synthetic Time-Series Generation for Household Power Consumption Using TimeGAN
* **Author / Researcher:** Gayatri 
* **Degree Program:** MSc in Data Analytics
* **Course:** Generative Artificial Intelligence Capstone Project

---

## 2. Problem Statement & Motivation

Sequential data—from industrial sensor streams and financial prices to smart-meter readings—drives critical tasks like forecasting, anomaly detection, and operational planning. Yet, gaining access to these datasets is often restricted by high collection costs or strict privacy regulations. Household electricity logs in particular can inadvertently expose sensitive personal details, such as home occupancy patterns and daily routines. 

Generative models capable of producing realistic synthetic time-series sequences offer a practical solution, allowing researchers and developers to build, test, and share algorithms without releasing or exposing original, sensitive records. Early recurrent GAN approaches for time series (such as C-RNN-GAN and RCGAN) rely strictly on adversarial supervision alone, failing to explicitly capture step-by-step temporal dynamics. This project implements **TimeGAN (Yoon et al., 2019)**, which solves this limitation by integrating an embedding network and a supervised next-step loss directly in latent space.

---

## 3. Dataset Description

* **Source:** UCI Machine Learning / Kaggle *Individual Household Electric Power Consumption* dataset.
* **Dataset Size:** Approximately $2.07\text{ million}$ raw per-minute measurements spanning 47 months.
* **Features ($D = 7$):**
  1. `Global_active_power`: Household global minute-averaged active power (kilowatts).
  2. `Global_reactive_power`: Household global minute-averaged reactive power (kilowatts).
  3. `Voltage`: Minute-averaged voltage (volts).
  4. `Global_intensity`: Household global minute-averaged current intensity (amperes).
  5. `Sub_metering_1`: Kitchen sub-metering (active energy in watt-hours of active power).
  6. `Sub_metering_2`: Laundry room sub-metering (watt-hours).
  7. `Sub_metering_3`: Electric water heater and air-conditioner sub-metering (watt-hours).
* **Target / Class Distribution:** Unsupervised multivariate time-series trajectory generation (continuous energy values rather than discrete class labels).

---

## 4. Methodology Overview (Lifecycle Stages)

The pipeline executes across a rigorous, end-to-end data science lifecycle:
1. **Data Cleaning & Resampling:** Raw per-minute telemetry undergoes anomaly filtering, missing-value linear interpolation, and hourly resampling to yield $34{,}589$ clean hourly records.
2. **Chronological Splitting & Scaling:** A strict $80/20$ chronological split prevents look-ahead data leakage. A Min–Max scaler is fitted exclusively on the training partition to map features into $[0, 1]$ for TimeGAN's sigmoid outputs.
3. **Sliding Window Slicing:** Overlapping 24-hour windows ($T = 24$, stride $1\text{ h}$) generate tensors of shape $(N, 24, 7)$, yielding $10{,}000$ training windows and $6{,}895$ test windows.
4. **Three-Phase TimeGAN Training:** 
   * *Phase 1:* Embedding network training via reconstruction loss ($\mathcal{L}_R$).
   * *Phase 2:* Supervised network training via next-step autoregressive loss ($\mathcal{L}_S$).
   * *Phase 3:* Joint adversarial training balancing unsupervised loss ($\mathcal{L}_U$) and supervised dynamics ($\mathcal{L}_S$).
5. **Inverse Transformation & Multi-Axis Evaluation:** Synthetic latent windows are generated and mapped back to physical units via inverse scaling for comprehensive evaluation.

---

## 5. Results Summary & Evaluation Metrics

We evaluate synthetic sequences across two primary axes:
* **Fidelity (Distributional & Temporal Resemblance):**
  * *Visual Inspection & Manifold Alignment:* PCA and t-SNE scatter plots demonstrate that synthetic trajectories successfully capture global operational manifolds.
  * *Correlation & Autocorrelation:* Feature cross-correlation matrices and Autocorrelation Function (ACF) decay curves confirm that diurnal cycles (morning rise at 06:00, evening peak at 20:00) and inter-variable relationships are well preserved.
* **Utility (Downstream Task Performance via TSTR):**
  * *Train on Synthetic, Test on Real (TSTR):* A post-hoc forecasting model trained on synthetic data achieves competitive predictive performance when evaluated on real held-out test data, confirming strong downstream utility.
* **Diagnostic Insights:** While marginal distributions and feature correlations align closely, the evaluation candidly highlights minor fidelity gaps in strict short-lag temporal dynamics and classifier indistinguishability.

---

## 6. Screenshots & Visualizations

*(Place your generated output plots inside an `outputs/` folder in your repository)*
* **Diurnal Cycle & Seasonality:** `outputs/figure1_patterns.png`
* **Feature Correlation Matrix:** `outputs/figure2_correlations.png`
* **PCA / t-SNE Manifold Comparison:** `outputs/pca_tsne_evaluation.png`
* **Autocorrelation (ACF) Curves:** `outputs/acf_comparison.png`

---

## 7. Local Setup & Installation Instructions

To set up and run this project locally on your machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/GAYATHRI0p0s/timegan-household-power.git](https://github.com/GAYATHRI0p0s/timegan-household-power.git)
   cd timegan-household-power
