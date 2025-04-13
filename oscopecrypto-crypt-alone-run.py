import os

# Environment configuration for a clean Jupyter environment with minimal logs and GPU setup.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "false"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_LOG_LEVEL"] = "ERROR"

# Set visible GPU devices before importing TensorFlow
USE_ONLY_GPU_INDEX = 0
from tensorflow.python.framework import config as tf_config

physical_devices = tf_config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf_config.set_visible_devices(physical_devices[USE_ONLY_GPU_INDEX], 'GPU')
        tf_config.set_memory_growth(physical_devices[USE_ONLY_GPU_INDEX], True)
    except RuntimeError as e:
        print("GPU config error:", e)
else:
    raise RuntimeError("No GPU found!")

# Import TensorFlow and silence warnings
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("GPUs available:", tf.config.list_logical_devices('GPU'))

import os
import time
import io
import re
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import load, dump
from backtesting import Backtest, Strategy
from bokeh.io import show, output_notebook, save, reset_output
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, MultiHeadAttention, Dropout, LayerNormalization, GlobalAveragePooling1D, Dense, Conv1D, MaxPooling1D, Flatten, GRU, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from datetime import datetime
import markdown
import io as stdio
from IPython.utils import io as ipy_io


ASSETS = ["btc", "eth"]
DATA_PERCENTAGES = percentages = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
EPOCHS = 10
STOP_LOSS = 0.003

PROCESSED_DATA = os.path.join("data", "processed-data")
RESULT_DATA_DIR = os.path.join("model-result-data")

os.makedirs(PROCESSED_DATA, exist_ok=True)
os.makedirs(RESULT_DATA_DIR, exist_ok=True)


# Data

def create_target_df(features_df, asset_label):
    target_df = pd.DataFrame(index=features_df.index)
    target_df[f'{asset_label}_Close_1m_later'] = features_df['Close'].shift(-1)
    target_df.dropna(inplace=True)
    return target_df

def extract_target_df(features_df, label = "target"):
    if label not in features_df.columns:
        raise ValueError(f"Label column '{label}' not found in DataFrame. Available columns: {list(features_df.columns)}")
    target_df = features_df[[label]].copy()
    features_df = features_df.drop(columns=[label])
    return features_df, target_df

def _clean(df, tz="UTC"):
    for c in ("minute_id", "Date_dt", "datetime", "timestamp"):
        if c in df.columns:
            df = df.rename(columns={c: "date"})
            break
    if pd.api.types.is_integer_dtype(df["date"]):
        df["date"] = pd.to_datetime(
            df["date"].astype(str), format="%Y%m%d%H%M", utc=True, errors="coerce"
        )
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.set_index("date").sort_index()
    return df.tz_convert(tz) if tz else df

def crypto_data_io(
    asset,
    action="load",
    fmt="auto",
    root=PROCESSED_DATA,
    tz="UTC",
    engine="pyarrow",
):
    asset = asset.lower()
    csv_file = os.path.join(root, f"{asset}_1min_with_features.csv.gz")
    pq_file = os.path.join(root, f"{asset}_1min_with_features.parquet")

    if fmt == "auto":
        fmt = "parquet" if os.path.exists(pq_file) else "csv"

    if action == "load":
        if fmt == "parquet":
            df = pd.read_parquet(pq_file, engine=engine)
        elif fmt == "csv":
            df = pd.read_csv(csv_file, compression="gzip")
        else:
            raise ValueError
        df = _clean(df, tz=tz)

    elif action == "save":
        df = _clean(pd.read_csv(csv_file, compression="gzip"), tz=tz)
        if fmt == "parquet":
            df.to_parquet(pq_file, index=True, compression="snappy", engine=engine)
        elif fmt == "csv":
            clean_csv = os.path.splitext(csv_file)[0] + ".clean.csv.gz"
            df.to_csv(clean_csv, compression="gzip")
        else:
            raise ValueError
    else:
        raise ValueError

    return df, create_target_df(df, asset)


btc_df_features, btc_targets = crypto_data_io("btc", action="load")
eth_df_features, eth_targets = crypto_data_io("eth", action="load")





# Constants & Functions

def setup_directories():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    return "Done"

def get_data_subset(features_df, targets_df, target_col, data_percent=0.01):
    # Intersect indices from features_df and targets_df, sorted
    common_index = features_df.index.intersection(targets_df.index).sort_values()
    
    subset_size = int(len(common_index) * data_percent)
    subset_dates = common_index[:subset_size]

    print("Total data values in intersection:", len(common_index))
    print("Number of data points to use:", subset_size)

    X = features_df.loc[subset_dates].copy()
    y = targets_df.loc[subset_dates, target_col].copy()

    # Create a combined subset dataframe
    subset = X.copy()
    subset[target_col] = y

    # Convert string index in the form YYYYMMDDHHMM to datetime
    date_series = pd.to_datetime(subset.index, format='%Y%m%d%H%M')

    subset.index = date_series
    X.index = date_series

    # Print diagnostic info
    print("Start of subset:", date_series[0] if not date_series.empty else "N/A")
    print("End of subset:", date_series[-1] if not date_series.empty else "N/A")

    return X, y, subset, date_series

def scale_features(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    indices = []  # will store i+time_steps for labeling
    for i in range(len(X) - time_steps):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps])
        indices.append(i + time_steps)
    return np.array(Xs), np.array(ys), np.array(indices)

def train_test_split_sequences(X_seq, y_seq, indices, train_ratio=0.8):
    train_size = int(len(X_seq) * train_ratio)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    idx_train = indices[:train_size]

    X_test  = X_seq[train_size:]
    y_test  = y_seq[train_size:]
    idx_test = indices[train_size:]

    return X_train, y_train, X_test, y_test, idx_train, idx_test

def format_percent_string(p):
    formatted = f"{p * 100:05.2f}"
    return formatted.replace(".", "_")


def train_model(model, X_train, y_train, X_val, y_val, data_percent=0.01, epochs=20, model_name="model", asset="btc"):
    
    percent_display = format_percent_string(data_percent)
    model_prefix = f"{model_name}_{asset.lower()}"
    
    model_filename = f"{model_prefix}_model_{percent_display}pct.h5"
    log_filename = f"{model_prefix}_log_{percent_display}pct.csv"

    checkpoint = ModelCheckpoint(os.path.join("models", model_filename), monitor="val_loss", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger(os.path.join("logs", log_filename))

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop, csv_logger],
        verbose=1
    )
    return history

def print_metrics(y_test, y_pred, *, data_percent=0.01,
                  model_name="model", dataset_name="asset"):
    import os, pandas as pd, numpy as np
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 mean_squared_error, mean_absolute_error)

    test_sig = np.sign(np.diff(y_test.ravel()))
    pred_sig = np.sign(np.diff(y_pred.ravel()))

    acc  = accuracy_score(test_sig, pred_sig)
    prec = precision_score(test_sig, pred_sig, average='macro')
    rec  = recall_score(test_sig, pred_sig, average='macro')
    f1   = f1_score(test_sig, pred_sig, average='macro')
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"{data_percent*100:.1f}% | acc {acc:.4f}  prec {prec:.4f}  "
          f"rec {rec:.4f}  f1 {f1:.4f}  mse {mse:.4f}  rmse {rmse:.4f}  mae {mae:.4f}")

    os.makedirs(RESULT_DATA_DIR, exist_ok=True)
    fname = f"{dataset_name}_{model_name}_{format_percent_string(data_percent)}pct_metrics.csv"
    pd.DataFrame([dict(dataset=dataset_name, model=model_name,
                       data_pct=data_percent, accuracy=acc, precision=prec,
                       recall=rec, f1=f1, mse=mse, rmse=rmse, mae=mae)]
                ).to_csv(os.path.join(RESULT_DATA_DIR, fname), index=False)

def plot_loss(history, plots_dir="plots", data_percent=0.01, model_name="gru", dataset_name="btc"):
    # Ensure the directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Format percent strings
    percent_display_filename = format_percent_string(data_percent) 
    percent_display_title = f"{data_percent * 100:.2f}"

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss Curve | Model: {model_name.upper()} | Dataset: {dataset_name.upper()} | Data: {percent_display_title}%")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)

    # Save plot
    filename = f"loss_curve_{model_name}_{dataset_name}_{percent_display_filename}pct.png"
    loss_curve_path = os.path.join(plots_dir, filename)
    plt.savefig(loss_curve_path)

    print(f"Loss curve saved to: {loss_curve_path}")

def plot_predictions(y_test, y_pred, date_series=None, plots_dir="plots", data_percent=0.01,
                     model_name="gru", dataset_name="btc", last_n=200):

    # Ensure directories exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(RESULT_DATA_DIR, exist_ok=True)

    # Fallback for date series
    if date_series is None:
        date_series = pd.RangeIndex(start=0, stop=len(y_test))

    # Format percent display
    percent_display_filename = format_percent_string(data_percent)
    percent_display_title = f"{data_percent * 100:.2f}"

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        "Date": date_series,
        "Actual": y_test,
        "Predicted": y_pred.flatten()
    })

    # Plot predictions
    plt.figure(figsize=(10, 4))
    plt.plot(comparison_df["Date"].iloc[-last_n:], comparison_df["Actual"].iloc[-last_n:], label="Actual")
    plt.plot(comparison_df["Date"].iloc[-last_n:], comparison_df["Predicted"].iloc[-last_n:], label="Predicted", linestyle="--")
    plt.title(f"{dataset_name.upper()} Prediction | Model: {model_name.upper()} | Data: {percent_display_title}% | Last {last_n} Samples")
    plt.xlabel("Date" if date_series is not None else "Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)

    # Save plot
    filename = f"prediction_plot_{model_name}_{dataset_name}_{percent_display_filename}pct.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    print(f"Prediction plot saved to: {plot_path}")

    # Save comparison CSV
    result_file = os.path.join(RESULT_DATA_DIR, f"{model_name}_{dataset_name}_{percent_display_filename}pct_comparison.csv")
    comparison_df.to_csv(result_file, index=False)
    print(f"Comparison CSV saved to: {result_file}")

    return comparison_df

def save_model_and_scaler(model, scaler, model_name: str, dataset_name: str, data_percent: float, model_ext: str = "h5"):
    os.makedirs("models", exist_ok=True)
    percent_str = f"{data_percent * 100:.4f}".rstrip("0").rstrip(".").replace(".", "_")
    model_filename = f"{model_name}_{dataset_name}_model_{percent_str}pct.{model_ext}"
    scaler_filename = f"{model_name}_{dataset_name}_scaler_{percent_str}pct.pkl"

    model_path = Path("models") / model_filename
    scaler_path = Path("models") / scaler_filename

    if model_ext in ['h5', 'keras']:
        save_model(model, str(model_path))
    else:
        dump(model, model_path)

    dump(scaler, scaler_path)

    print("Model saved to:", model_path.resolve().relative_to(Path.cwd()))
    print("Scaler saved to:", scaler_path.resolve().relative_to(Path.cwd()))

def load_model_and_scaler(model_name: str, dataset_name: str, data_percent: float, model_ext: str = "h5"):
    percent_str = f"{data_percent * 100:.4f}".rstrip("0").rstrip(".").replace(".", "_")

    model_filename = f"{model_name}_{dataset_name}_model_{percent_str}pct.{model_ext}"
    scaler_filename = f"{model_name}_{dataset_name}_scaler_{percent_str}pct.pkl"

    model_path = Path("models") / model_filename
    scaler_path = Path("models") / scaler_filename

    if model_ext in ['h5', 'keras']:
        model = load_model(model_path)
    else:
        model = load(model_path)

    scaler = load(scaler_path)

    print("Model loaded from:", model_path.resolve().relative_to(Path.cwd()))
    print("Scaler loaded from:", scaler_path.resolve().relative_to(Path.cwd()))

    return model, scaler

class MLStrategy(Strategy):
    stop_loss_pct = 0.0
    def init(self):
        self.signal = self.data.Signal
        num_pos_signals = int((self.signal == 1).sum())
        num_neg_signals = int((self.signal == -1).sum())
        print(f"Found {num_pos_signals} +1 signals and {num_neg_signals} -1 signals in the data.")
        self.opened_first_time = False

    def next(self):
        # Get the current bar's index and use it to fetch the full data (OHLC and Signal) from the original DataFrame.
        current_idx = self.data.index[-1]
        current_close = self.data.df.loc[current_idx, 'Close']
        current_signal = self.data.df.loc[current_idx, 'Signal']
        current_profit = 0.0
        if self.position and hasattr(self, 'my_entry_price'):
            if self.position.is_long:
                current_profit = current_close - self.my_entry_price
            elif self.position.is_short:
                current_profit = self.my_entry_price - current_close
    
        # 1) For the first trade: only enter when the signal is +1.
        if not self.opened_first_time:
            if current_signal == 1:
                self.opened_first_time = True
                if self.stop_loss_pct > 0:
                    stop_loss_price = current_close * (1 - self.stop_loss_pct)
                    self.buy(sl=stop_loss_price)
                else:
                    self.buy()
                # Record our entry price for subsequent profit calculations.
                self.my_entry_price = current_close
            return  # Exit early until first trade is taken.
    
        # 2) For subsequent bars, adjust positions based on the signal.
        if current_signal == 1:
            # If not currently long, flip to long (close any short and go long).
            if not self.position.is_long:
                self.position.close()
                if self.stop_loss_pct > 0:
                    stop_loss_price = current_close * (1 - self.stop_loss_pct)
                    self.buy(sl=stop_loss_price)
                else:
                    self.buy()
                # Update the entry price after the trade is executed.
                self.my_entry_price = current_close
    
        elif current_signal == -1:
            # Only reverse to short if there's an open long position that is already in profit.
            if self.position and self.position.is_long and current_profit > 0:
                self.position.close()
                if self.stop_loss_pct > 0:
                    stop_loss_price = current_close * (1 + self.stop_loss_pct)
                    self.sell(sl=stop_loss_price)
                else:
                    self.sell()
                # Update the entry price for the new short position.
                self.my_entry_price = current_close

    def stop(self):
        # Force-close any open position at the end of the backtest
        self.position.close()

def save_interactive_chart(bt_, html_filename):
    with ipy_io.capture_output() as captured:
        plot_obj = bt_.plot()
    save(plot_obj, filename=html_filename)

def run_backtest(
    df_bt,
    initial_cash=10_000,
    model_name="gru",
    dataset_name="btc",
    data_percent=0.01,
    stop_loss_pct=0.0
):
    print("Setting up Backtest object...")
    bt = Backtest(
        df_bt,
        MLStrategy,
        cash=initial_cash,
        commission=0.0,
        exclusive_orders=True
    )
    
    print("Running backtest engine...")
    stats = bt.run(stop_loss_pct=stop_loss_pct)
    print(stats)
    
    trades_df = getattr(stats, 'trades', None) or getattr(stats, '_trades', None)
    if trades_df is not None:
        n_trades = len(trades_df)
        n_closed_trades = trades_df['ExitPrice'].notna().sum()
        n_open_trades = trades_df['ExitPrice'].isna().sum()
        
        print("\nTrade Details:")
        print(f"Number of trades:       {n_trades}")
        print(f"Number of closed trades:{n_closed_trades}")
        print(f"Number of open trades:  {n_open_trades}")
    
    print("Saving backtest plot to HTML...")
    from bokeh.io import save
    import os
    
    # Format stop loss as percentage string for filenames
    stop_str = format_percent_string(stop_loss_pct)  # e.g. 0.025 becomes 'sl0025'
    result_dir = os.path.join(
        "results",
        f"{model_name}_{dataset_name}_{format_percent_string(data_percent)}pct_{stop_str}stloss"
    )
    os.makedirs(result_dir, exist_ok=True)
    #have to save this outside, as this was making the notebook crash
    html_filename = os.path.join(result_dir, "backtest_plot.html")
    save_interactive_chart(bt, html_filename)
    csv_filename = os.path.join(result_dir, "backtest_stats.csv")
    stats.to_csv(csv_filename)
    print("Backtest stats saved to:", csv_filename)
    return stats

# Optomized Funtions

def train_and_predict(asset, model_function, data_percent, model_name=None, epochs=20, time_steps=30, other_df=None):
    print(f"Loading data for asset: {asset}")
    if asset.lower() == "btc":
        features_df = btc_df_features
        targets_df = btc_targets
        target_col = "btc_Close_1m_later"
    elif asset.lower() == "eth":
        features_df = eth_df_features
        targets_df = eth_targets
        target_col = "eth_Close_1m_later"
    elif other_df is not None and not other_df.empty:
        features_df, targets_df = extract_target_df(other_df, 'target')
        target_col = "target"
    else:
        raise ValueError("Asset must be either 'btc' or 'eth' or a df should be given.")
    
    print("Subsetting data...")
    X_raw, y_raw, merged_subset, date_series = get_data_subset(features_df, targets_df, target_col, data_percent)
    print(f"Data subset shape: {X_raw.shape}, {y_raw.shape}")
    df_ohlc = merged_subset[["Open", "High", "Low", "Close"]].copy()
    X_no_ohlc = X_raw.drop(columns=["Open", "High", "Low", "Close"], errors='ignore')
    
    print("Normalizing features...")
    X_scaled, scaler = scale_features(X_no_ohlc)
    
    print("Creating sequences...")
    X_seq, y_seq, all_indices = create_sequences(X_scaled, y_raw.values, time_steps)
    print(f"Total sequences: {len(X_seq)}")
    
    print("Splitting data into training and testing sets...")
    X_train, y_train, X_test, y_test, idx_train, idx_test = train_test_split_sequences(X_seq, y_seq, all_indices)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = model_function(input_shape)
    
    if not model_name:
        model_name = model_function.__name__ if hasattr(model_function, '__name__') else "custom_model"
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, data_percent, epochs, model_name=model_name, asset=asset)
    
    print("Plotting training loss...")
    plot_loss(history, data_percent=data_percent, model_name=model_name, dataset_name=asset)
    
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")
    
    print("Plotting predictions and saving comparison CSV...")
    test_dates = date_series[idx_test]
    
    plot_predictions(y_test, y_pred, date_series=test_dates, data_percent=data_percent, model_name=model_name, dataset_name=asset)
    
    print("Printing evaluation metrics...")
    print_metrics(y_test, y_pred, data_percent=data_percent, model_name=model_name, dataset_name=asset)
    
    print("Saving model and scaler...")
    save_model_and_scaler(model, scaler, model_name=model_name, dataset_name=asset, data_percent=data_percent)
    
    print("Training and prediction complete.")
    
    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "df_ohlc": df_ohlc,
        "idx_test": idx_test,
        "date_series": date_series,      # In the initial implementation and runs, we did not include date_series; this was done later, and we ran the CNN BTC for this to test.
        "model_name": model_name,
        "asset_name": asset
    }

def run_backtest_pipeline(
    df_ohlc, 
    y_pred, 
    idx_test,
    initial_cash=100_000, 
    model_name="gru", 
    dataset_name="btc", 
    data_percent=0.02, 
    stop_loss_pct=0.0
):
    df_ohlc_test = df_ohlc.iloc[idx_test].copy()
    df_ohlc_test.reset_index(drop=True, inplace=True)
    # Create a simple long/short signal based on whether we think next price is above current close:
    df_ohlc_test["Signal"] = np.where(
        y_pred.flatten() > df_ohlc_test["Close"].values,
        1,
        -1
    )
    
    print("Signal distribution:")
    print(df_ohlc_test["Signal"].value_counts())
    
    stats= run_backtest(
        df_bt=df_ohlc_test,
        initial_cash=initial_cash,
        model_name=model_name,
        dataset_name=dataset_name,
        data_percent=data_percent,
        stop_loss_pct=stop_loss_pct
    )
    
    return stats

def run_full_pipeline(
    asset, 
    model_function, 
    data_percent, 
    model_name=None, 
    epochs=20, 
    time_steps=30, 
    initial_cash=100_000, 
    stop_loss_pct=0.0,
    other_df=None
):
    
    training_outputs = train_and_predict(
        asset=asset,
        model_function=model_function,
        data_percent=data_percent,
        model_name=model_name,
        epochs=epochs,
        time_steps=time_steps,
        other_df = other_df
    )
    
    df_ohlc = training_outputs["df_ohlc"]
    y_pred = training_outputs["y_pred"]
    idx_test = training_outputs["idx_test"]
    
    stats = run_backtest_pipeline(
        df_ohlc=df_ohlc,
        y_pred=y_pred,
        idx_test=idx_test,
        initial_cash=initial_cash,
        model_name=training_outputs["model_name"],
        dataset_name=training_outputs["asset_name"],
        data_percent=data_percent,
        stop_loss_pct=stop_loss_pct
    )

    print("Full pipeline complete.")
    return stats

# Models

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_transformer_model(input_shape):
    print("Building Transformer model with input shape:", input_shape)
    input_layer = Input(shape=input_shape)
    attention = MultiHeadAttention(num_heads=2, key_dim=input_shape[-1])(input_layer, input_layer)
    attention = Dropout(0.2)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + input_layer)
    pooled = GlobalAveragePooling1D()(attention)
    dense = Dense(64, activation='relu')(pooled)
    output = Dense(1, activation='linear')(dense)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def build_rnn_model(input_shape):
    print("Building RNN model with input shape:", input_shape)
    model = Sequential()
    model.add(SimpleRNN(64, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Full run


def execute_experiment_for_percentage(asset, model_name, model_function, data_pct, epochs, stop_loss, md_base):
    lines = []  
    start = time.time()
    percent_str = format_percent_string(data_pct)
    
    # Capture stdout during run_full_pipeline (and any nested prints)
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        stats = run_full_pipeline(
            asset=asset,
            model_name=model_name,
            epochs=epochs,
            model_function=model_function,
            data_percent=data_pct,
            stop_loss_pct=stop_loss
        )
    captured_output = stdout_buffer.getvalue()
    # Remove backspace control characters while preserving newlines and normal characters.
    captured_output = re.sub(r'\x08+', '', captured_output)
    duration = time.time() - start

    # Compute artifact file paths.
    training_image_path = os.path.join("plots", f"loss_curve_{model_name}_{asset}_{percent_str}pct.png")
    prediction_plot_path = os.path.join("plots", f"prediction_plot_{model_name}_{asset}_{percent_str}pct.png")
    
    # Use prediction CSV (comparison) instead of the old epoch CSV.
    prediction_csv_path = os.path.join("model-result-data", f"{model_name.lower()}_{asset.lower()}_comparison_{percent_str}pct.csv")
    # Compute the metrics CSV path (as generated in print_metrics).
    metrics_csv_path = os.path.join("model-result-data", f"{asset.lower()}_{model_name.lower()}_{percent_str}pct_metrics.csv")
    stop_str = format_percent_string(stop_loss)
    result_folder = os.path.join(
        "results",
        f"{model_name}_{asset}_{format_percent_string(data_pct)}pct_{stop_str}stloss"
        )
    backtest_html_path = os.path.join(result_folder, "backtest_plot.html")
    backtest_csv_path = os.path.join(result_folder, "backtest_stats.csv")
    model_file = os.path.join("models", f"{model_name}_{asset}_model_{percent_str}pct.h5")
    scaler_file = os.path.join("models", f"{model_name}_{asset}_scaler_{percent_str}pct.pkl")
    
    # Utility: Compute relative paths (using forward slashes).
    def rel_path(p):
        return os.path.relpath(p, start=md_base).replace(os.path.sep, '/')

    # Begin constructing the Markdown snippet.
    lines.append(f"#### Experiment: Data Used: {data_pct:.3%}, Epochs: {epochs}")
    lines.append(f"**Stop Loss:** {stop_loss:.3%}\n")
    lines.append(f"**Duration:** {duration:.2f} seconds\n")
    
    # Console output as a subheading.
    if captured_output.strip():
        lines.append("##### Console Output:")
        lines.append("```\n" + captured_output + "\n```")
    
    # Training Loss Plot as a subheading.
    if os.path.exists(training_image_path):
        lines.append("##### Training Loss Plot:")
        lines.append(f"![Training Loss Plot]({rel_path(training_image_path)})\n")
    
    # Prediction Plot as a subheading.
    if os.path.exists(prediction_plot_path):
        lines.append("##### Prediction Plot:")
        lines.append(f"![Prediction Plot]({rel_path(prediction_plot_path)})\n")
    
    # Comparison CSV link as a subheading.
    if os.path.exists(prediction_csv_path):
        lines.append("##### Comparison Data:")
        lines.append(f"[View CSV]({rel_path(prediction_csv_path)})\n")
    
    # Metrics CSV link as a subheading.
    if os.path.exists(metrics_csv_path):
        lines.append("##### Metrics CSV:")
        lines.append(f"[View CSV]({rel_path(metrics_csv_path)})\n")
    
    # Backtest Interactive Chart as a subheading.
    lines.append("##### Backtest Interactive Chart:")
    if os.path.exists(backtest_html_path):
        # Embed the HTML as an iframe.
        iframe_html = f'<iframe src="{rel_path(backtest_html_path)}" width="100%" height="600px" frameborder="0"></iframe>'
        lines.append(iframe_html)
        lines.append(f"If the iframe does not display, please [click here to view the HTML]({rel_path(backtest_html_path)}).\n")
    else:
        lines.append(f"Backtest interactive chart HTML file not found at {backtest_html_path}\n")
    
    # Backtest Stats as a subheading.
    if os.path.exists(backtest_csv_path):
        lines.append("##### Backtest Stats:")
        try:
            df_csv = pd.read_csv(backtest_csv_path)
            csv_md = df_csv.to_markdown(index=False)
            lines.append(csv_md + "\n")
        except Exception:
            lines.append(f"[View CSV]({rel_path(backtest_csv_path)})\n")
    
    # Saved Model and Scaler as a subheading.
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        lines.append("##### Saved Model and Scaler:")
        lines.append(f"- **Model:** [View Model]({rel_path(model_file)})")
        lines.append(f"- **Scaler:** [View Scaler]({rel_path(scaler_file)})\n")
    
    # Combine all lines into the final markdown snippet.
    markdown_snippet = "\n".join(lines) + "\n\n---\n\n"
    
    # Save the markdown (including the captured output) into a dedicated text file.
    text_log_dir = os.path.join("logs", "experiment_text")
    os.makedirs(text_log_dir, exist_ok=True)
    text_log_file = os.path.join(text_log_dir, f"experiment_{asset}_{model_name}_{percent_str}pct.txt")
    with open(text_log_file, "w") as log_file:
        log_file.write(markdown_snippet)
    
    # Return useful information along with the generated markdown snippet.
    return {
        "stats": stats,
        "markdown": markdown_snippet,
        "comparison_csv": prediction_csv_path,
        "metrics_csv": metrics_csv_path,
        "text_log": text_log_file
    }

def execute_model_runs(models_dict, assets, data_percentages, epochs=10, stop_loss=0.003, display_report=True):
    results = {}
    csv_paths = []
    report_filenames = []
    report_dir = os.path.join("logs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Loop over each data percentage to create individual reports
    for data_pct in data_percentages:
        report_md_lines = []
        pct_str = format_percent_string(data_pct)
        start_time = time.time()
        
        # Build report header using a more descriptive title.
        report_md_lines.append(f"# Full Analysis Report for Data Percentage: {data_pct * 100:.2f}%\n")
        report_md_lines.append("This report summarizes all experiments conducted for this specific data percentage.\n")
        report_md_lines.append(f"**Assets Tested:** {', '.join(assets)}\n")
        report_md_lines.append(f"**Epochs per Experiment:** {epochs}\n")
        report_md_lines.append(f"**Stop Loss Setting:** {stop_loss}\n")
        report_md_lines.append("---\n")
        
        for asset in assets:
            # Initialize results for the asset if not already present
            if asset not in results:
                results[asset] = {}
            report_md_lines.append(f"\n## Asset: {asset.upper()}\n")
            
            for model_name, model_fn_list in models_dict.items():
                if model_name not in results[asset]:
                    results[asset][model_name] = {}
                report_md_lines.append(f"\n### Model: {model_name.upper()}\n")
                
                for model_fn in model_fn_list:
                    # Execute the experiment for current asset/model/data percentage.
                    exp_result = execute_experiment_for_percentage(
                        asset=asset,
                        model_name=model_name,
                        model_function=model_fn,
                        data_pct=data_pct,
                        epochs=epochs,
                        stop_loss=stop_loss,
                        md_base=report_dir
                    )
                    
                    # Remove unwanted experiment run heading from the returned markdown snippet.
                    exp_markdown_lines = [
                        line for line in exp_result["markdown"].splitlines()
                        if not line.startswith("#### Experiment: Data Used:")
                    ]
                    cleaned_markdown = "\n".join(exp_markdown_lines)
                    
                    # Store the stats in the results dictionary.
                    results[asset][model_name][data_pct] = exp_result["stats"]
                    
                    # Add the cleaned experiment markdown to the report content.
                    report_md_lines.append(cleaned_markdown)
                    
                    # Collect the comparison and metrics CSV paths.
                    csv_paths.append({
                        "asset": asset,
                        "model": model_name,
                        "data_pct": data_pct,
                        "comparison_csv": exp_result["comparison_csv"],
                        "metrics_csv": exp_result["metrics_csv"]
                    })
        
        total_duration = time.time() - start_time
        report_md_lines.append(f"\n---\n**Total Duration for Data Percentage {data_pct * 100:.2f}%:** {total_duration:.2f} seconds\n")
        full_md_content = "\n".join(report_md_lines)
        
        # Create report filename using the formatted percentage.
        report_filename = f"experiment_report_{pct_str}pct.md"
        md_filename = os.path.join(report_dir, report_filename)
        report_filenames.append(md_filename)
        
        with open(md_filename, "w") as f:
            f.write(full_md_content)
        
        if display_report:
            print(full_md_content)
            print(f"\nMarkdown report saved to: {md_filename}\n")
            try:
                display(Markdown(full_md_content))
            except Exception:
                pass

    return results, report_filenames, csv_paths

def conv_md_toc(report_path):
    html_file = os.path.splitext(report_path)[0] + '.html'
    with open(report_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    lines = md_content.splitlines()
    spaced = []
    for i, line in enumerate(lines):
        spaced.append(line)
        if i < len(lines) - 1 and lines[i].strip() and lines[i+1].strip():
            spaced.append("")
    toc = ("<div style='text-align: center; font-weight: bold; font-size: 1.5em; "
           "font-family: Georgia, serif;'>Table of Content</div>\n\n[TOC]\n\n")
    new_lines = []
    toc_inserted = False
    for line in spaced:
        if not toc_inserted and line.strip() == '---':
            new_lines.append(toc.strip())
            new_lines.append("")
            toc_inserted = True
        new_lines.append(line)
    md_with_toc = "\n".join(new_lines) if toc_inserted else toc + "\n".join(spaced)
    html_body = markdown.markdown(
        md_with_toc,
        extensions=['toc', 'fenced_code', 'tables'],
        extension_configs={'toc': {'permalink': True, 'toc_depth': '2-6', 'title': ''}}
    )
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Report</title>
    <style>
        body {{
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 16px;
            line-height: 1.6;
            padding: 40px;
            max-width: 800px;
            margin: auto;
        }}
        h1, h2, h3, h4, h5 {{ font-weight: bold; }}
        ul {{ margin-left: 1.5em; }}
        code, pre {{
            font-family: Consolas, monospace;
            background: #f4f4f4;
            padding: 4px;
            border-radius: 4px;
        }}
        .toc ul {{ list-style-type: decimal; padding-left: 20px; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    return html_file

def load_logs(log_dir="logs"):
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".csv")]
    summary_data = []
    all_logs = []

    for path in log_files:
        df = pd.read_csv(path)
        filename = os.path.basename(path).lower()

        model_match = re.match(r'(.*?)_(btc|eth)_log_(\d+_\d+|\d+)pct\.csv', filename)
        if not model_match:
            continue

        model_type = model_match.group(1).upper()
        crypto = model_match.group(2).upper()
        data_pct_str = model_match.group(3)
        data_pct = float(data_pct_str.replace("_", "."))

        min_val_loss = df['val_loss'].min()
        min_val_epoch = df['val_loss'].idxmin()
        final_val_loss = df['val_loss'].iloc[-1]
        epochs_run = len(df)

        summary_data.append({
            "Type": model_type,
            "Crypto": crypto,
            "Data %": data_pct,
            "Min Val Loss": min_val_loss,
            "Epoch (Min Val Loss)": min_val_epoch,
            "Final Val Loss": final_val_loss,
            "Epochs Run": epochs_run,
            "File": filename
        })

        df['Model'] = model_type
        df['Crypto'] = crypto
        df['Data %'] = data_pct
        df['File'] = filename
        all_logs.append(df)

    if all_logs:
        logs_df = pd.concat(all_logs, ignore_index=True)
    else:
        logs_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_data).sort_values(by=["Type", "Crypto", "Data %"])
    
    available_pcts = sorted(summary_df["Data %"].unique())
    print("Available Data Percentages:", available_pcts)

    # Save compiled CSVs in log_dir
    summary_path = os.path.join(log_dir, "compiled_summary.csv")
    logs_path = os.path.join(log_dir, "compiled_logs.csv")
    
    summary_df.to_csv(summary_path, index=False)
    logs_df.to_csv(logs_path, index=False)
    
    print(f"Saved summary to {summary_path}")
    print(f"Saved logs to {logs_path}")

    return summary_df, logs_df, available_pcts


def plot_loss_val_loss(logs_df, available_pcts, selected_pct='all', save_dir="logs"):
    if logs_df.empty:
        raise ValueError("logs_df is empty. Nothing to plot.")

    # Determine which percentages to plot
    if selected_pct == 'all':
        pcts_to_plot = available_pcts
    elif selected_pct in available_pcts:
        pcts_to_plot = [selected_pct]
    else:
        raise ValueError(f"Selected data percentage {selected_pct}% not found in logs. Available options: {available_pcts}")

    palette = sns.color_palette("tab10")
    model_types = logs_df['Model'].unique()
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(model_types)}

    for pct in pcts_to_plot:
        grouped = logs_df[logs_df["Data %"] == pct].groupby(['Crypto'])

        for (crypto,), group in grouped:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

            # LOSS subplot
            ax = axes[0]
            for model in model_types:
                model_data = group[group['Model'] == model].sort_values('epoch')
                if not model_data.empty:
                    ax.plot(model_data['epoch'], model_data['loss'], label=model, color=color_map[model], linewidth=2)
            ax.set_title(f"Training Loss - {crypto} - {pct}% Data")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()

            # VAL_LOSS subplot
            ax = axes[1]
            for model in model_types:
                model_data = group[group['Model'] == model].sort_values('epoch')
                if not model_data.empty:
                    ax.plot(model_data['epoch'], model_data['val_loss'], label=model, color=color_map[model], linewidth=2, linestyle='--')
            ax.set_title(f"Validation Loss - {crypto} - {pct}% Data")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Loss")
            ax.grid(True)
            ax.legend()

            plt.tight_layout()

            # Save figure
            formatted_pct = format_percent_string(pct/100)
            filename = f"loss_curve_all_{crypto.lower()}_{formatted_pct}pct.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            print(f"Saved plot to: {save_path}")


def conv_all(report_paths):
    html_files = []
    for rpt in report_paths:
        hf = conv_md_toc(rpt)
        print(f"Converted {rpt} -> {hf}")
        html_files.append(hf)
    return html_files

def main():

        




    models = {
        "gru": [build_gru_model],
        "cnn": [build_cnn_model],
        "transformer": [build_transformer_model],
        "rnn": [build_rnn_model],
    }



    results, report_filenames, csv_paths = execute_model_runs(
        models_dict=models,
        assets=ASSETS,
        data_percentages=DATA_PERCENTAGES,
        epochs=EPOCHS,
        stop_loss=STOP_LOSS,
        display_report=False
    )

    print("Final Experiment Report Generated.\n")
    print("Your comprehensive experiment report (with summaries, logs, and artifact links) has been created.\n")
    print("Report Links:")

    for i, path in enumerate(report_filenames, start=1):
        print(f"  {i}. {path}")


    converted_html = conv_all(report_filenames)
    print("Converted HTML files:")

    for html_file in converted_html:
        print(html_file)

    print("report_filenames:", report_filenames)

    print("csv_paths:", csv_paths)



    summary_df, logs_df, available_pcts = load_logs("logs")
    summary_df


    plot_loss_val_loss(logs_df, available_pcts, selected_pct='all')




if __name__ == "__main__":
    main()