#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import glob
import os
import time

###############################################################################
# CONFIG
###############################################################################

RESOURCE_DIR = "data/MSResource/*.csv"
RT_DIR       = "data/MSRTQps/*.csv"

CHUNK_SIZE   = 300_000          # Safe for 2.5GB CSVs
PART_SIZE    = 500_000          # Number of samples per output file
OUTPUT_DIR   = "processed_parts"
FINAL_DATASET = "alibaba_dataset.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# STREAM CSV LOADER
###############################################################################

def stream_csv(path, chunksize=CHUNK_SIZE):
    """Stream huge CSV file safely."""
    return pd.read_csv(path, chunksize=chunksize)

###############################################################################
# FIRST PASS: Identify the top microservice globally
###############################################################################

def find_top_microservice(res_files, rt_files):
    """
    Stream all pairs of Resource + RT CSVs and find which microservice
    has the highest average RPS globally.
    """
    print("Scanning data to find top microservice...")

    rps_accumulator = {}

    for res_path, rt_path in zip(res_files, rt_files):

        print(f"  → scanning pair: {os.path.basename(res_path)}")

        for res_chunk, rt_chunk in zip(stream_csv(res_path),
                                       stream_csv(rt_path)):

            # sanitize columns
            res_chunk.columns = res_chunk.columns.str.lower().str.strip()
            rt_chunk.columns  = rt_chunk.columns.str.lower().str.strip()

            # compute rps per (timestamp, msname)
            df_rps = (
                rt_chunk[rt_chunk['metric']
                         .str.contains("providerrpc_mcr", case=False, na=False)]
                .groupby(['timestamp','msname'])['value']
                .mean()
                .reset_index()
            )

            # accumulate rps totals for averaging
            for _, row in df_rps.iterrows():
                ms = row["msname"]
                val = float(row["value"])
                if ms not in rps_accumulator:
                    rps_accumulator[ms] = {"sum": 0.0, "count": 0}
                rps_accumulator[ms]["sum"]   += val
                rps_accumulator[ms]["count"] += 1

    # compute global average RPS per service
    averages = {
        ms: vals["sum"] / vals["count"]
        for ms, vals in rps_accumulator.items()
        if vals["count"] > 0
    }

    top_ms = max(averages, key=averages.get)
    print(f"\n✅ Top microservice identified: {top_ms}\n")
    return top_ms

###############################################################################
# SECOND PASS: Build dataset ONLY for top microservice
###############################################################################

def build_dataset(res_files, rt_files, top_ms):
    """
    Stream through the data files again and generate imitation-learning samples
    for the specified microservice.
    Writes output files in chunks.
    """
    print("Building dataset for microservice:", top_ms)

    dataset_part = []
    part_id = 0
    total_count = 0

    def save_part():
        nonlocal dataset_part, part_id
        if not dataset_part:
            return
        out_path = os.path.join(OUTPUT_DIR, f"dataset_part_{part_id}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dataset_part, f)
        print(f"✅ Saved {len(dataset_part)} samples → {out_path}")
        dataset_part = []
        part_id += 1

    for res_path, rt_path in zip(res_files, rt_files):

        print(f"  → processing pair: {os.path.basename(res_path)}")

        for res_chunk, rt_chunk in zip(stream_csv(res_path),
                                       stream_csv(rt_path)):

            res_chunk.columns = res_chunk.columns.str.lower().str.strip()
            rt_chunk.columns  = rt_chunk.columns.str.lower().str.strip()

            # --- replica counts ---
            replicas = (
                res_chunk.groupby(['timestamp','msname'])['msinstanceid']
                .nunique()
                .reset_index()
                .rename(columns={'msinstanceid':'replica_count'})
            )

            # --- avg cpu + memory ---
            state_res = (
                res_chunk.groupby(['timestamp','msname'])
                [['cpu_utilization','memory_utilization']]
                .mean()
                .reset_index()
            )

            # --- rps ---
            df_rps = (
                rt_chunk[rt_chunk['metric']
                         .str.contains("providerrpc_mcr", case=False, na=False)]
                .groupby(['timestamp','msname'])['value']
                .mean()
                .reset_index()
                .rename(columns={'value':'rps'})
            )

            # --- merge ---
            merged = (
                state_res.merge(replicas, on=['timestamp','msname'])
                         .merge(df_rps, on=['timestamp','msname'], how='left')
                         .fillna(0)
            )

            # filter for OUR chosen microservice only
            merged = merged[merged["msname"] == top_ms]
            if merged.empty:
                continue

            # --- Build samples ---
            for _, row in merged.iterrows():
                obs = np.zeros(16, dtype=np.float32)

                obs[0] = row['cpu_utilization']
                obs[1] = row['memory_utilization']
                obs[5] = row['rps'] / 500.0
                obs[9] = row['replica_count'] / 20.0
                obs[10] = obs[9]
                obs[11] = 1.0

                # Action is the expert replica count (bounded 1–10)
                replicas_val = int(min(max(row['replica_count'],1),10)) - 1

                dataset_part.append((obs, replicas_val))
                total_count += 1

                # write part to disk
                if len(dataset_part) >= PART_SIZE:
                    save_part()

    # save remaining
    save_part()

    print(f"\n✅ Finished building dataset for {top_ms}")
    print(f"✅ Total samples: {total_count}\n")
    return total_count

###############################################################################
# FINAL MERGE: Combine part files into one final dataset
###############################################################################

def merge_parts():
    print("Merging dataset parts into a final .pkl ...")
    dataset = []

    part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "dataset_part_*.pkl")))

    for p in part_files:
        with open(p, "rb") as f:
            dataset.extend(pickle.load(f))
        print(f"  → loaded {p}")

    with open(FINAL_DATASET, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\n✅ Final dataset saved: {FINAL_DATASET}")
    print(f"✅ Total samples: {len(dataset)}")

###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":

    res_files = sorted(glob.glob(RESOURCE_DIR))
    rt_files  = sorted(glob.glob(RT_DIR))

    if len(res_files) != len(rt_files):
        print("❌ Mismatch: number of MSResource and MSRTQps files differ!")
        exit(1)

    top_ms = find_top_microservice(res_files, rt_files)
    build_dataset(res_files, rt_files, top_ms)
    merge_parts()

    print("\n✅ ALL DONE.")
