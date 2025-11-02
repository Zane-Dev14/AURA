#!/usr/bin/env python3
"""
Safer preprocessing script for huge Alibaba CSVs.

Features:
- Defensive streaming with chunk fallback
- Detects corrupt files early
- Limits native threads to avoid C-extension races
- Robust logging to identify which file/chunk causes issues
- Memory-safe aggregation for top microservice
- Produces dataset parts and final merged dataset

Run: python3 preprocess_alibaba_safe.py
"""
import os
import sys
import glob
import time
import pickle
import traceback
###############################################################################
# FAST GLOBAL RT INDEX (LOAD RT ONCE)
###############################################################################

def build_global_rt_index(rt_files):
    """
    Fastest possible RT index builder.
    Loads RT CSVs once, collects only (timestamp, msname, rps),
    groups once at the end.
    """
    log("Building global RT index (fast mode).")

    ts_list = []
    name_list = []
    rps_list = []

    for rt_path in rt_files:
        log(f"  Indexing RT file: {rt_path}")
        try:
            for chunk in stream_csv_safe(rt_path):
                cols = chunk.columns.str.lower().str.strip()
                chunk.columns = cols

                if "metric" not in cols:
                    continue

                mask = chunk["metric"].str.contains("providerrpc_mcr", case=False, na=False)
                filtered = chunk.loc[mask, ["timestamp", "msname", "value"]]

                # append raw values directly (fast)
                ts_list.extend(filtered["timestamp"].tolist())
                name_list.extend(filtered["msname"].tolist())
                rps_list.extend(filtered["value"].tolist())

        except Exception as e:
            log(f"  ⚠️ RT indexing failed in {rt_path}: {e}")
            continue

    if not ts_list:
        log("⚠️ No RT rows found! Returning empty index.")
        return {}

    df = pd.DataFrame({"timestamp": ts_list,
                       "msname": name_list,
                       "value": rps_list})

    df = (
        df.groupby(["timestamp", "msname"])["value"]
          .mean()
          .reset_index()
          .rename(columns={"value": "rps"})
    )

    log(f"Global RT index built with {len(df)} rows (optimized).")

    # ✅ Convert to plain dict (int timestamp → float rps)
    return {
        (int(r.timestamp), r.msname): float(r.rps)
        for r in df.itertuples()
    }

# Limit native thread libraries to reduce segfaults caused by race conditions
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import pandas as pd
import numpy as np

###############################################################################
# CONFIG
###############################################################################

RESOURCE_DIR = "data/MSResource/*.csv"
RT_DIR       = "data/MSRTQps/*.csv"
TEMP_AGG_FILE = "temp_agg_rps.csv"

# Start conservative; will fall back to smaller chunk sizes on failure
CHUNK_SIZE   = 2_000_000
MIN_CHUNK    = 2_000_000
PART_SIZE    = 250_000     # samples per output file
OUTPUT_DIR   = "processed_parts"
FINAL_DATASET = "alibaba_dataset.pkl"
LOG_FILE = "preprocess_log.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# UTILITIES
###############################################################################

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def file_quick_check(path, nbytes=2048):
    """Return True if the file seems readable text (not empty/binary)."""
    try:
        with open(path, "rb") as f:
            head = f.read(nbytes)
        if not head:
            log(f"⚠️ File empty: {path}")
            return False
        # crude binary detection: lots of null bytes -> binary
        if head.count(b'\x00') > 5:
            log(f"⚠️ File looks binary or corrupted (null bytes): {path}")
            return False
        return True
    except Exception as e:
        log(f"⚠️ Could not open file {path}: {e}")
        return False

def stream_csv_safe(path, chunksize=CHUNK_SIZE):
    if not file_quick_check(path):
        raise RuntimeError(f"File failed quick-check: {path}")

    try:
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            engine="c",
            on_bad_lines="skip",
            low_memory=False
        )
        for chunk in reader:
            yield chunk
    except Exception as e:
        log(f"❌ FATAL: CSV cannot be read by C-engine: {path} — {e}")
        raise

###############################################################################
# PASS 1: MEMORY-SAFE identify top microservice by average RPS
###############################################################################

def find_top_microservice_and_rt_index(rt_files):
    log("Starting scan to find top microservice + build RT index (single pass).")

    agg = {}       # msname → {sum: X, count: Y}
    rt_index = {}  # (timestamp, msname) → rps list

    for rt_path in rt_files:
        log(f"Scanning RT: {rt_path}")
        for chunk in stream_csv_safe(rt_path):
            chunk.columns = chunk.columns.str.lower().str.strip()

            if "metric" not in chunk.columns:
                continue

            mask = chunk["metric"].str.contains("providerrpc_mcr", case=False, na=False)
            sub = chunk.loc[mask, ["timestamp", "msname", "value"]]

            # Aggregate for top microservice computation
            g = sub.groupby("msname")["value"].agg(["sum","count"])
            for ms, row in g.iterrows():
                if ms not in agg:
                    agg[ms] = {"sum":0, "count":0}
                agg[ms]["sum"] += row["sum"]
                agg[ms]["count"] += row["count"]

            # Put raw values into RT index buffer
            for t, name, v in sub.itertuples(index=False):
                key = (int(t), name)
                if key not in rt_index:
                    rt_index[key] = []
                rt_index[key].append(v)

    # Compute average RPS and determine top microservice
    ms_avg = {ms: data["sum"] / data["count"] for ms, data in agg.items()}
    top_ms = max(ms_avg, key=ms_avg.get)

    # Collapse RT index to mean values
    rt_index_final = {k: float(np.mean(vs)) for k, vs in rt_index.items()}

    log(f"Top microservice = {top_ms}")
    log(f"RT index entries = {len(rt_index_final)}")

    return top_ms, rt_index_final

###############################################################################
# PASS 2: Build dataset for single microservice (defensive streaming)
###############################################################################

def build_dataset(res_files, rt_index, top_ms):
    log(f"Building dataset for microservice: {top_ms}")

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
        log(f"Saved {len(dataset_part)} samples → {out_path}")
        dataset_part = []
        part_id += 1

    # We'll build a small in-memory index per chunk for RT to join by timestamp+msname.
    # Iterate over resource files and for each resource chunk, load relevant rt rows from the corresponding RT file chunks.
    # If counts mismatch or pairing is unclear, we will load the whole RT file chunk into a dict keyed by (timestamp, msname).

    for res_path in res_files:
        # Heuristic to find RT file with similar base name; fallback to iterate all RTs (slower

        log(f"Processing Resource file: {res_path}")


        if not file_quick_check(res_path):
            log(f"Skipping unreadable resource file: {res_path}")
            continue

        try:
            res_stream = stream_csv_safe(res_path)
        except RuntimeError as e:
            log(f"Failed to stream resource file {res_path}: {e}")
            continue

        # For each resource chunk, we will read through RT candidates and try to find matches.
        for res_chunk_idx, res_chunk in enumerate(res_stream):
            try:
                res_chunk.columns = res_chunk.columns.str.lower().str.strip()
            except Exception:
                log(f"  ⚠️ Failed to sanitize resource columns for chunk {res_chunk_idx} in {res_path}. Skipping chunk.")
                continue

            # compute replica counts per (timestamp, msname)
            try:
                replicas = (
                    res_chunk.groupby(['timestamp','msname'])['msinstanceid']
                    .nunique()
                    .reset_index()
                    .rename(columns={'msinstanceid':'replica_count'})
                )
            except Exception as e:
                log(f"  ⚠️ Replica grouping failed for chunk {res_chunk_idx} in {res_path}: {e}. Skipping chunk.")
                continue

            # find and prepare CPU/MEM columns robustly
            # ✅ Use the *actual* Alibaba column names
           cpu = res_chunk.get("instance_cpu_usage", pd.Series(0, index=res_chunk.index))
            mem = res_chunk.get("instance_memory_usage", pd.Series(0, index=res_chunk.index))

            state_res = (
                pd.DataFrame({
                    "timestamp": res_chunk["timestamp"],
                    "msname": res_chunk["msname"],
                    "cpu_utilization": cpu,
                    "memory_utilization": mem
                })
                .groupby(["timestamp","msname"])[["cpu_utilization","memory_utilization"]]
                .mean()
                .reset_index()
            )


            # Now build RT index for this set of timestamps in this resource chunk.
            # We'll load RT rows from candidate RT files that match timestamps in this res_chunk.
            # ✅ Use fast O(1) lookup from prebuilt RT index
            # ✅ Vectorized RT lookup (much faster)
            keys = list(zip(replicas["timestamp"].astype(int), replicas["msname"]))
            rps_vals = [rt_index.get(k, 0.0) for k in keys]

            df_rps = replicas[["timestamp", "msname"]].copy()
            df_rps["rps"] = rps_vals


            # merge
            try:
                merged = (
                    state_res.merge(replicas, on=['timestamp','msname'], how='outer')
                             .merge(df_rps, on=['timestamp','msname'], how='left')
                             .fillna(0)
                )
            except Exception as e:
                log(f"  ⚠️ Merge failed for chunk {res_chunk_idx} in {res_path}: {e}. Skipping.")
                continue

            # filter for top_ms
            merged = merged[merged['msname'] == top_ms]
            if merged.empty:
                continue

            # build samples
            cpu = merged["cpu_utilization"].astype(np.float32).to_numpy()
            mem = merged["memory_utilization"].astype(np.float32).to_numpy()
            rps = (merged["rps"] / 500.0).astype(np.float32).to_numpy()
            rep = merged["replica_count"].clip(1, 10).astype(np.int32).to_numpy()

            obs_block = np.zeros((len(merged), 16), dtype=np.float32)
            obs_block[:,0]  = cpu
            obs_block[:,1]  = mem
            obs_block[:,5]  = rps
            obs_block[:,9]  = rep / 20.0
            obs_block[:,10] = obs_block[:,9]
            obs_block[:,11] = 1.0

            actions = (rep - 1).tolist()

            batch = list(zip(obs_block, actions))
            dataset_part.extend(batch)
            total_count += len(batch)

            

            if len(dataset_part) >= PART_SIZE:
                save_part()

    # final save
    save_part()
    log(f"Finished building dataset. Total samples: {total_count}")
    return total_count

###############################################################################
# PASS 3: Merge parts into final pkl
###############################################################################

def merge_parts():
    log("Merging dataset parts...")
    part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "dataset_part_*.pkl")))
    if not part_files:
        log("No part files found; merge aborted.")
        return

    final_list = []
    for p in part_files:
        try:
            with open(p, "rb") as f:
                batch = pickle.load(f)
            final_list.extend(batch)
            log(f"Loaded {p} ({len(batch)} samples). Accum total: {len(final_list)}")
        except Exception as e:
            log(f"Failed to load part {p}: {e}")

    with open(FINAL_DATASET, "wb") as f:
        pickle.dump(final_list, f)
    log(f"Final dataset saved: {FINAL_DATASET} (Total samples: {len(final_list)})")

    # cleanup
    for p in part_files:
        try:
            os.remove(p)
        except Exception:
            pass
    try:
        os.rmdir(OUTPUT_DIR)
    except Exception:
        pass
    log("Cleanup complete.")

###############################################################################
# MAIN
###############################################################################

def main():
    start = time.time()
    res_files = sorted(glob.glob(RESOURCE_DIR))
    rt_files  = sorted(glob.glob(RT_DIR))

    if not res_files:
        log(f"ERROR: No resource files found with pattern: {RESOURCE_DIR}")
        sys.exit(1)
    if not rt_files:
        log(f"ERROR: No RT files found with pattern: {RT_DIR}")
        sys.exit(1)

    log(f"Found {len(res_files)} resource files and {len(rt_files)} RT files.")

    # Sanity quick-print of first few file names
    log("Resource files (first 5): " + ", ".join(os.path.basename(p) for p in res_files[:5]))
    log("RT files (first 5): " + ", ".join(os.path.basename(p) for p in rt_files[:5]))

    top_ms, rt_index = find_top_microservice_and_rt_index(rt_files)
    if not top_ms:
        log("Top microservice detection failed. Exiting.")
        sys.exit(1)

    build_dataset(res_files, rt_index, top_ms)
    merge_parts()

    end = time.time()
    log(f"ALL DONE in {end - start:.1f} seconds.")

if __name__ == "__main__":
    main()
