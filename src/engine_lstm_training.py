"""LSTM seq2seq training driver.

For each climate division on the CLI, for each NYSM station in that
division, for each supported variable (`t2m`, `u_total`, `tp`), and
for each HRRR forecast hour 1..18 (visited in random order):

1. Build the training dataframe with `prepare_lstm_data.prepare_lstm_data`.
2. Wrap it in `SequenceDatasetMultiTask` (per-window z-score, NYSM
   persistence on the future portion of the encoder input).
3. Train `encode_decode_lstm.ShallowLSTM_seq2seq_multi_task` with the
   `OutlierFocusedLoss` and the `ReduceLROnPlateau` scheduler.
4. Persist the encoder + decoder weights for the
   `(climdiv, metvar, station)` combination to `MODEL_DIR`.

CLI
---
    python engine_lstm_training.py --clim_div "Hudson Valley" --device_id 0
    python engine_lstm_training.py --clim_div "Eastern Plateau" "Western Plateau" --device_id 1
"""

import argparse
import gc
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append("..")

from model_architecture import encode_decode_lstm, sequencer
from model_data import hrrr_data, nysm_data, prepare_lstm_data


print("imports loaded")


# Override these via environment variables if your filesystem layout
# differs from the defaults.
MODEL_DIR = os.environ.get(
    "LSTM_MODEL_DIR",
    "/home/aevans/inference_ai2es_forecast_err/MODELS",
)


def custom_collate(batch):
    """Drop `None` items the dataset may emit (e.g. all-zero precip
    sequences) before delegating to the default collate."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def date_filter(ldf, time1, time2):
    """Strict date-window filter (`time1 < valid_time < time2`)."""
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]
    return ldf


class EarlyStopper:
    """Plain-vanilla patience-based early stopping.

    Keeps track of the best loss seen so far; once `patience`
    consecutive epochs have failed to improve on it (by more than
    `min_delta`), `early_stop` returns True.
    """

    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class OutlierFocusedLoss(nn.Module):
    """MAE loss that up-weights high-magnitude errors.

    `loss = mean( (|err| + 1)^alpha * |err| )`

    `alpha = 0` reduces to plain MAE; larger `alpha` makes the loss
    increasingly sensitive to outliers, which is useful when the
    target distribution is heavy-tailed (like NWP forecast error).
    """

    def __init__(self, alpha, device):
        super().__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        error = y_true - y_pred
        base_loss = torch.abs(error)
        weights = (torch.abs(error) + 1).pow(self.alpha)
        return (weights * base_loss).mean()


def get_model_file_size(file_path):
    """Print the on-disk size of a saved checkpoint in MB."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")


def save_model_weights(model, encoder_path, decoder_path):
    """Persist encoder + decoder state dicts to disk."""
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)


def main(
    start_time,
    end_time,
    batch_size,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    device,
    hrrr_df,
    nwp_model="HRRR",
    sequence_length=30,
    target="target_error",
    learning_rate=5e-5,
    save_model=True,
):
    """Train one model per `(station, metvar)` in the given climate
    division for forecast hour `fh`.

    Parameters
    ----------
    start_time, end_time : datetime
        Inclusive start / exclusive end of the training window.
    batch_size, num_layers, epochs, weight_decay, learning_rate : numeric
        Standard hyper-parameters.
    fh : int
        Forecast hour (1..18) being trained.
    clim_div : str or list[str]
        Climate division name(s) (will be space-joined if a list).
    device : torch.device
        Compute device.
    hrrr_df : pandas.DataFrame
        Pre-loaded HRRR forecasts for `fh`.
    nwp_model : str
        Identifier of the NWP model the data is sourced from.
    sequence_length : int
        Length of the past window fed to the encoder.
    target : str
        Name of the column predicted by the model.
    learning_rate : float
        AdamW learning rate.
    save_model : bool
        If True, persist final weights even if the best-on-train check
        never triggered (defensive).
    """
    print("CUDA available?", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if not isinstance(device, torch.device):
        device = torch.device(device)
    print("Requested device:", device)
    # CPU-only PyTorch builds (common on macOS) cannot call CUDA APIs even when a
    # cuda:* device was requested — fall back to CPU on any failure.
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_id = device.index if device.index is not None else 0
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        except (AttributeError, AssertionError, RuntimeError) as exc:
            print(f"Warning: CUDA not usable ({exc!r}); using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    torch.manual_seed(101)
    print("::: LSTM Training :::")

    # Use the start_time's year to choose which year of NYSM data to load.
    year = start_time.year
    if isinstance(clim_div, list):
        clim_div = " ".join(clim_div)
    print("Climate division:", clim_div)

    nysm_df = nysm_data.load_nysm_data(year)
    clim_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    clim_df_filt = clim_df[clim_df["climate_division_name"] == clim_div]

    stations_in_div = clim_df_filt["stid"].unique().tolist()
    print("Stations in division:", stations_in_div)

    for stid in stations_in_div:
        print("Station:", stid)
        filtered_df = date_filter(nysm_df, start_time, end_time)
        hrrr_df_filt = date_filter(hrrr_df, start_time, end_time)

        for metvar in ["t2m", "u_total", "tp"]:
            decoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_decoder.pth"
            )
            encoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_encoder.pth"
            )

            # Build the training dataframe for this station / metvar.
            (
                lstm_df,
                features,
                stations,
                target,
                valid_times,
            ) = prepare_lstm_data.prepare_lstm_data(
                filtered_df, hrrr_df_filt, stid, metvar, fh=fh, train=True
            )
            print(f"Features: {len(features)}")
            print(f"Target: {target}")

            train_dataset = sequencer.SequenceDatasetMultiTask(
                dataframe=lstm_df,
                target=target,
                features=features,
                sequence_length=sequence_length,
                forecast_steps=fh,
                device=device,
                metvar=metvar,
            )

            train_kwargs = {
                "batch_size": batch_size,
                "pin_memory": False,
                "shuffle": True,
                "collate_fn": custom_collate,
            }
            train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            print("Data loader ready.")

            num_sensors = int(len(features))
            hidden_units = int(12 * len(features))

            # Single shared encoder + decoder; the multi-task name is
            # historical (the original architecture supported per-station
            # decoder heads).
            model = encode_decode_lstm.ShallowLSTM_seq2seq_multi_task(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=num_layers,
                mlp_units=1500,
                device=device,
                num_stations=len(stations),
            ).to(device)

            # Warm-start from previous weights if they exist.
            if os.path.exists(encoder_path):
                print("Loading existing encoder weights")
                model.encoder.load_state_dict(
                    torch.load(encoder_path), strict=False
                )
                get_model_file_size(encoder_path)

            if os.path.exists(decoder_path):
                print("Loading existing decoder weights")
                model.decoder.load_state_dict(
                    torch.load(decoder_path), strict=False
                )
                get_model_file_size(decoder_path)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            loss_function = OutlierFocusedLoss(2.0, device)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=4
            )

            print("--- Training LSTM ---")
            early_stopper = EarlyStopper(10)
            train_loss_ls = []
            for ix_epoch in range(1, epochs + 1):
                gc.collect()
                train_loss = model.train_model(
                    data_loader=train_loader,
                    loss_func=loss_function,
                    optimizer=optimizer,
                    epoch=ix_epoch,
                    training_prediction="recursive",
                    teacher_forcing_ratio=0.5,
                )
                scheduler.step(train_loss)
                train_loss_ls.append(train_loss)
                if early_stopper.early_stop(train_loss):
                    print(f"Early stopping at epoch {ix_epoch}")
                    break
                # Save whenever the running min improves (after a brief
                # warmup so we don't checkpoint random first epochs).
                if train_loss <= min(train_loss_ls) and ix_epoch > 5:
                    print(f"Saving model weights at epoch {ix_epoch}")
                    save_model_weights(model, encoder_path, decoder_path)
                    save_model = False

            # Defensive final save in case the running-min check never
            # fired (e.g. epochs <= 5 or non-decreasing loss curve).
            if save_model:
                save_model_weights(model, encoder_path, decoder_path)

            print("... completed ...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clim_div", nargs="+", required=True, help="Climate division name(s)"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="GPU device id (e.g. 0 for cuda:0)",
    )
    args = parser.parse_args()
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    now = datetime.now()
    year = now.year

    # Training window.  Pick something appropriate to the years of HRRR
    # parquets that exist on disk.
    start_time = datetime(2018, 10, 1, 0, 0, 0)
    end_time = datetime(2025, 5, 5, 23, 59, 0)

    # Train each forecast hour in random order so that any short-circuit
    # failure leaves a representative subset of `fh`s trained.
    fh_all = np.arange(1, 19)
    fh = fh_all.copy()
    while len(fh) > 0:
        fh_r = int(random.choice(fh))
        try:
            print(f"-- Loading HRRR data for FH {fh_r} --")
            hrrr_df = hrrr_data.read_hrrr_data(str(fh_r).zfill(2), year)
            main(
                start_time=start_time,
                end_time=end_time,
                batch_size=1000,
                num_layers=3,
                epochs=50,
                weight_decay=0.0,
                fh=fh_r,
                clim_div=args.clim_div,
                device=device,
                hrrr_df=hrrr_df,
            )
        except Exception as e:
            print(f"-- ERROR for FH {fh_r}: {e} --")
        gc.collect()
        fh = fh[fh != fh_r]
