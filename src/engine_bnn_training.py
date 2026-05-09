"""BNN training driver.

Mirrors `engine_lstm_training.py` exactly except:
    * model is `bnn.ShallowLSTM_seq2seq_multi_task_bnn`
    * the loss is Gaussian NLL (computed INSIDE
      `bnn.ShallowLSTM_seq2seq_multi_task_bnn.train_model`); we still
      pass an `OutlierFocusedLoss` to satisfy the function signature
      but it is ignored
    * we save three weight files per (station, metvar, fh):
      encoder, decoder, and the bnn head
"""

import sys

sys.path.append("..")

import argparse
import gc
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_architecture import bnn, sequencer
from model_data import hrrr_data, nysm_data, prepare_lstm_data


print("imports loaded")


MODEL_DIR = os.environ.get(
    "BNN_MODEL_DIR",
    "/home/aevans/inference_ai2es_forecast_err/MODELS/BNN",
)


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]
    return ldf


class EarlyStopper:
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


class _Placeholder(nn.Module):
    """Placeholder loss func - the BNN computes Gaussian NLL itself."""

    def forward(self, *_args, **_kwargs):
        raise RuntimeError("BNN computes its own loss internally.")


def _save_weights(model, encoder_path, decoder_path, bnn_path):
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    torch.save(model.bnn.state_dict(), bnn_path)


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
    learning_rate=5e-5,
    save_model=True,
):
    print("CUDA available?", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())
    if not isinstance(device, torch.device):
        device = torch.device(device)
    print("Requested device:", device)
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
    print(" *********")
    print("::: BNN Training :::")

    year = start_time.year
    if isinstance(clim_div, list):
        clim_div = " ".join(clim_div)
    print(clim_div)

    nysm_df = nysm_data.load_nysm_data(year)
    clim_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    clim_df_filt = clim_df[clim_df["climate_division_name"] == clim_div]

    stations_in_div = clim_df_filt["stid"].unique().tolist()
    print(stations_in_div)
    for stid in stations_in_div:
        print(stid)
        filtered_df = date_filter(nysm_df, start_time, end_time)
        hrrr_df_filt = date_filter(hrrr_df, start_time, end_time)

        for metvar in ["t2m", "u_total", "tp"]:
            encoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_encoder.pth"
            )
            decoder_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_decoder.pth"
            )
            bnn_path = (
                f"{MODEL_DIR}/{clim_div}_{metvar}_{stid}_fh{fh}_bnn.pth"
            )

            (
                lstm_df,
                features,
                stations,
                target,
                valid_times,
            ) = prepare_lstm_data.prepare_lstm_data(
                filtered_df, hrrr_df_filt, stid, metvar, fh=fh, train=True
            )
            print("FEATURES", len(features))
            print("TARGET", target)

            train_dataset = sequencer.SequenceDatasetMultiTask(
                dataframe=lstm_df,
                target=target,
                features=features,
                sequence_length=sequence_length,
                forecast_steps=fh,
                device=device,
                metvar=metvar,
                nwp_model=nwp_model,
            )

            train_kwargs = {
                "batch_size": batch_size,
                "pin_memory": False,
                "shuffle": True,
                "collate_fn": custom_collate,
            }
            train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            print("!! Data Loaders Successful !!")

            num_sensors = int(len(features))
            hidden_units = int(12 * len(features))

            model = bnn.ShallowLSTM_seq2seq_multi_task_bnn(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=num_layers,
                mlp_units=1500,
                device=device,
                num_stations=len(stations),
                seq_len=sequence_length,
                input_dim=num_sensors,
            ).to(device)

            if os.path.exists(encoder_path):
                print("Loading Encoder Model")
                model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
            if os.path.exists(decoder_path):
                print("Loading Decoder Model")
                model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
            if os.path.exists(bnn_path):
                print("Loading BNN Head")
                model.bnn.load_state_dict(torch.load(bnn_path), strict=False)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=4
            )

            early_stopper = EarlyStopper(10)
            print("--- Training BNN ---")

            train_loss_ls = []
            for ix_epoch in range(1, epochs + 1):
                gc.collect()
                train_loss = model.train_model(
                    data_loader=train_loader,
                    loss_func=_Placeholder(),
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
                if train_loss <= min(train_loss_ls) and ix_epoch > 5:
                    print(f"Saving BNN weights... EPOCH {ix_epoch}")
                    _save_weights(model, encoder_path, decoder_path, bnn_path)
                    save_model = False

            if save_model:
                _save_weights(model, encoder_path, decoder_path, bnn_path)

            print("... completed ...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clim_div", nargs="+", required=True, help="List of climate divisions"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="Device to use (e.g., 0 for cuda:0)",
    )
    args = parser.parse_args()
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    now = datetime.now()
    year = now.year

    start_time = datetime(2018, 10, 1, 0, 0, 0)
    end_time = datetime(2025, 5, 5, 23, 59, 0)

    nwp_model = "HRRR"
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
