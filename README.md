# O-Scope Crypto: Price & Sentiment Predictor

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Authors:** [Zain Ali](https://github.com/zainnobody), [Hallie Kinsey](https://github.com/halliekinsey), and [Nick Short](https://github.com/nshort2001)


Explore the full project on the [GitHub repository](https://github.com/zainnobody/capstone-project).  
The dataset is available on [Hugging Face](https://huggingface.co/datasets/zainnobody/capstone-project-data),  
and interactive reports and results can be viewed on the [project documentation site](https://zainnobody.github.io/capstone-project/docs).


---

## Overview

**O-Scope Crypto** is a capstone project that integrates high-frequency cryptocurrency market data with sentiment analysis derived from financial news. By merging engineered technical indicators with sentiment scores from FinBERT, the project designs an ensemble of deep learning models (GRU, CNN, RNN, and Transformer) to predict short-term price changes for Bitcoin (BTC) and Ethereum (ETH). The pipeline also features robust backtesting mechanisms to evaluate the proposed trading signals under realistic market conditions.

Key objectives include:
- **Data Fusion:** Synchronizing minute-level price data with aggregated financial news sentiment.
- **Feature Engineering:** Computing technical indicators such as moving averages, RSI, MACD, Bollinger Bands, and additional volume and candlestick metrics.
- **Modeling:** Training multiple deep learning architectures that capture different temporal nuances.
- **Backtesting:** Evaluating the performance of trading strategies generated from model predictions while incorporating risk management (stop-loss).

This project bridges traditional technical analysis with real-time sentiment analytics to help investors decipher the volatile and dynamic cryptocurrency landscape.

---

## Documentation & Presentation

- **Final Project Report:** Detailed documentation and an essay outlining background, data discovery, feature engineering, modeling strategies, results, and conclusions can be found in `oscopecrypto-final.pdf`.
- **Web Documentation:** A live documentation site is available in the [`docs/`](docs/) folder, including pages on model architectures, reports, and results.
- **Presentation Materials:** Supplementary slides and dashboards are hosted under the `docs/models/` and `docs/reports.html` pages.

---

## Repository Structure

The repository is organized to separate data, models, logs, results, and documentation clearly. Below is an annotated tree view and description of the key files and directories:

```
O-Scope Crypto/
├── LICENSE                         # MIT License file.
├── README.md                       # This README file.
├── oscopecrypto-final.pdf          # Final project report (PDF) with detailed explanation and results.
├── oscopecrypto-final.html         # HTML version of the final report.
├── oscopecrypto-final.ipynb        # Main Jupyter Notebook with the complete pipeline.
├── assets/                         # Static assets for styling and interactivity.
│   ├── css/                        # Cascading style sheets for web documentation.
│   └── js/                         # JavaScript files to enhance dashboard functionality.
├── data/
│   └── processed-data/             # All processed data files and features.
│       ├── btc_1min_processed.csv.gz           # Raw 1-minute BTC price data post processing.
│       ├── eth_1min_processed.csv.gz           # Raw 1-minute ETH price data post processing.
│       ├── crypto_1min_combined.csv.gz         # Combined BTC and ETH price data.
│       ├── aggregated-news_filtered.csv.gz     # Cleaned and time-aligned news article data.
│       ├── aggregated_news_with_features.parquet  # News sentiment features with various aggregations.
│       ├── aggregated_realistic_news_with_features_filtered.parquet  # Final filtered news features.
│       ├── btc_1min_with_features.parquet       # BTC data merged with engineered technical indicators.
│       ├── eth_1min_with_features.parquet       # ETH data merged with engineered technical indicators.
│       └── btc_test_predictions_with_features.csv.gz  # Test set predictions for BTC for evaluation.
├── docs/
│   ├── index.html                  # Main documentation landing page.
│   ├── dashboard.html              # Dashboard for viewing trends, model metrics, and reports.
│   ├── models/
│   │   ├── cnn.html                # Detailed description of the CNN model architecture and performance.
│   │   ├── gru.html                # Detailed description of the GRU model architecture and performance.
│   │   ├── rnn.html                # Detailed description of the RNN model architecture and performance.
│   │   └── transformer.html        # Detailed description of the Transformer model architecture and performance.
│   ├── reports.html                # Summaries and full reports of experiments.
│   └── results.html                # Visualized outcomes and charts from backtesting.
├── logs/                          # Logs and experiment artifacts.
│   ├── compiled_logs.csv           # Aggregated logs from training runs.
│   ├── compiled_summary.csv        # Summary statistics for various runs.
│   └── experiment_text/            # Text logs from experiments (one file per experiment).
├── model-result-data/             # CSV files containing metrics, prediction comparisons, and model evaluation results.
│   ├── cnn_btc_10pct_comparison.csv     # Prediction vs. actual comparisons for CNN on BTC.
│   ├── gru_eth_10pct_metrics.csv          # Metrics for GRU on ETH.
│   └── (other similar files for different models and data percentages)
├── models/                        # Saved model checkpoints and corresponding scalers.
│   ├── cnn_btc_model_10pct.h5      # Trained CNN model for BTC (10% data sample).
│   ├── gru_eth_model_10pct.h5      # Trained GRU model for ETH (10% data sample).
│   ├── transformer_btc_scaler_10pct.pkl  # Scaler used for Transformer model normalization.
│   └── (other models and scalers with similar naming conventions)
├── plots/                         # Generated plots and visualizations (loss curves, prediction charts, confusion matrices, etc.)
│   ├── loss_curve_cnn_btc_10pct.png  # Training loss curve for CNN on BTC (10% data sample).
│   ├── prediction_plot_gru_eth_10pct.png  # Prediction vs actual plot for GRU on ETH.
│   └── (additional plots for each experiment)
├── results/                       # Backtest results, interactive HTML charts, and summary CSV files.
│   ├── cnn_btc_10pct_0_3stloss/    # Folder containing backtest HTML report and CSV stats for CNN on BTC with a 0.3% stop-loss.
│   └── (other folders for different models, assets, data samples, and stop-loss settings)
├── requirements.txt               # Python dependencies and packages required.
```

### File Naming Conventions

- **Data Files (in `data/processed-data/`)**  
  - Files like `btc_1min_processed.csv.gz` and `eth_1min_processed.csv.gz` contain raw price data for BTC and ETH after preprocessing.  
  - Files with names like `btc_1min_with_features.parquet` have been augmented with technical indicators, enabling richer feature sets for model training.
  
- **Model Checkpoints (in `models/`)**  
  - Model files include the model type, asset, and the percentage of data used. For example, `cnn_btc_model_10pct.h5` is the CNN model trained on 10% of the BTC data sample.
  - Scaler files (with `.pkl` extension) are saved alongside their respective models to ensure identical data normalization during inference.

- **Logs and Results (in `logs/` and `model-result-data/`)**  
  - Log files such as `cnn_btc_log_10pct.csv` and experiment text files inside `logs/experiment_text/` capture training progression and hyperparameter configurations.
  - CSV files in `model-result-data/` contain performance metrics (e.g., RMSE, accuracy, R²) and prediction comparisons used for further analysis.

- **Plots (in `plots/`)**  
  - Loss curves, prediction plots, confusion matrices, and other diagnostics are saved with names that include the model type, asset, data percentage, and sometimes stop-loss settings.

- **Results Directory (in `results/`)**  
  - Each folder in `results/` corresponds to a specific experiment setup (e.g., model type, asset, data percentage, and stop-loss parameter). They include interactive backtest charts (HTML) and CSV files summarizing trade performance.

---

## Features

- **Integrated Pipeline:**  
  - Data ingestion from multiple sources: minute-level price data and large-scale financial news.
  - Automated feature engineering including technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) and sentiment variables (article count, sentiment scores, rolling sentiment).
  
- **Deep Learning Models:**  
  - Implements four separate architectures—GRU, CNN, RNN, and Transformer—to capture different temporal and volatility aspects.
  - Ensemble predictions provide robust trading signals for short-term forecasts.
  
- **Backtesting Module:**  
  - A custom backtesting strategy evaluates the practical efficacy of model predictions with built-in risk management (e.g., stop-loss mechanisms).
  
- **Comprehensive Logging and Reporting:**  
  - Extensive logs and experiment files allow for detailed performance tracking and reproducibility.
  - Generated Markdown/HTML reports in the `logs/reports/` folder summarize all experiments.

---

## Installation & Dependencies

**Prerequisites:**
- Python 3.8 or later.
- Jupyter Notebook (for exploring and running `.ipynb` files).
- Required Python libraries: TensorFlow, Keras, scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Bokeh, joblib, transformers, among others.
- For backtesting: the `Backtesting.py` package.

**Installation Steps:**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/zainnobody/capstone-project.git
   cd capstone-project
   ```
2. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If any package is missing, please install it manually using pip.)*

3. **Launch the Main Notebook:**
   ```bash
   jupyter notebook oscopecrypto-final.ipynb
   ```
4. **Explore the Documentation:**
   - View model details and reports on the local docs pages (open `docs/index.html` in your browser).

---

## Usage

- **Running the Pipeline:**  
  Open the `oscopecrypto-final.ipynb` notebook to run through the complete pipeline—from data ingestion and feature engineering to model training and backtesting.
  
- **Backtesting & Analysis:**  
  Examine the results generated in the `results/` folder. Interactive backtest charts (HTML) and CSV summaries provide detailed performance metrics.

- **Experiment Logs:**  
  Review logs under `logs/` and experiment text files within `logs/experiment_text/` for insights into training progress, hyperparameters, and system messages.

- **Dashboard & Reports:**  
  The `docs/dashboard.html` page and `logs/reports/` directory hold full experiment reports with embedded plots, metrics, and analysis summaries.

---

## Model Performance & Evaluation

Performance and metrics for each model are stored under:
- **`model-result-data/`**: Contains CSV files with evaluation metrics (accuracy, RMSE, R², etc.).
- **`plots/`**: Contains visualizations such as loss curves and prediction comparison plots.
- **`results/`**: Each subfolder corresponds to an experiment (e.g., `cnn_btc_10pct_0_3stloss`) with detailed backtesting outcomes.

The report also explains how file names embed vital information (model type, asset, data sample percentage, and stop-loss parameters) to help you quickly identify and compare experiment results.

---

## Limitations & Future Work

- **Prototype Status:** O-Scope Crypto is a research prototype and requires further validation before production use.
- **Data & Computational Constraints:** The current implementation uses capped data percentages to manage computational load; scaling up will be a focus for future work.
- **Integration Enhancements:** Plans include transitioning from historical batch processing to real-time data feeds and expanding sentiment analysis with additional data sources (e.g., social media).
- **Model Optimization:** Further optimization of deep learning models (e.g., hyperparameter tuning and architecture modifications) is necessary for live deployment.

---

## References

- CNN Business. (2025). *Fear & Greed Index*. CNN. https://www.cnn.com/markets/fear-and-greed
- CoinMarketCap. (2025). *Cryptocurrency Market Capitalization*. https://coinmarketcap.com/
- Fosso Wamba, S. et al. (2020). *Bitcoin, Blockchain and Fintech: a systematic review and case studies in the supply chain*. Production Planning & Control.
- Fidelity Investments. (2025). *Active Trader Pro Overview*. https://www.fidelity.com/
- Kottarathil, P. (2024). *Ethereum Historical Dataset*. Kaggle. https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset
- Zielinski, M. (2024). *Bitcoin Historical Data*. Kaggle. https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
- Zhao, D. (2024). *FNSPID: A Fine-Grained Financial News Dataset for Public Investors*. GitHub. https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- [Additional references detailed in the final report.]

---

## License

This project is licensed under the [MIT License](./LICENSE).