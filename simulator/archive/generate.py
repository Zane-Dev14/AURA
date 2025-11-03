#!/usr/bin/env python3
"""
preprocess_alibaba_make_big_dataset.py (FIXED v3 - Production Safe)

Goal:
- Produce a HIGH-QUALITY, BALANCED, and LARGE dataset.
- Uses a 3-Pass streaming architecture to handle 5GB+ of files on 16GB RAM.
- Pass 1: Memory-safe scan to find Top-K services.
- Pass 2: Build a *filtered* in-memory index for only Top-K services.
- Pass 3: Stream resource files, augment, and build the dataset.
- We will filter "junk" (idle) rows.
- We will *strategically oversample* rare actions (like scaling to 10 pods)
  and *undersample* common actions (like staying at 1 pod).
"""
import os
import sys
import time
import glob
import csv
import pickle
from collections import defaultdict, Counter
import math

# limit native threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import numpy as np
import pandas as pd

# --------------------- CONFIG ---------------------
RESOURCE_GLOB = "data/MSResource/*.csv"
RT_GLOB       = "data/MSRTQps/*.csv"
TEMP_AGG_FILE = "temp_agg_rps.csv" # For Pass 1

RT_CHUNK_ROWS = 1_000_000
TOP_K         = 50
PART_SIZE     = 500_000        # 500k samples per part file
OUTPUT_DIR    = "processed_parts_v3" # New folder
FINAL_DATASET = "alibaba_dataset_large.pkl" # New name
LOG_FILE      = "preprocess_log_v3.txt" # New log

# --- NEW QUALITY SETTINGS ---
# We no longer care about total bytes. We care about *good* samples.
MIN_RPS_FOR_AUGMENT = 10.0  # Don't augment "idle" data
MAX_SAMPLES_HARD = 15_000_000  # 15M samples is ~1.1GB. Perfect for 16GB RAM.

RNG_SEED = 123456
# ----------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def file_quick_check(path, nbytes=2048):
    try:
        with open(path, "rb") as f:
            head = f.read(nbytes)
        if not head:
            log(f"⚠️ File empty: {path}")
            return False
        if head.count(b'\x00') > 5:
            log(f"⚠️ File looks binary/corrupt: {path}")
            return False
        return True
    except Exception as e:
        log(f"⚠️ Cannot open file {path}: {e}")
        return False

# ------------------- PASS 1: Find Top-K (Memory-Safe) -------------------
def find_top_k_services_safe(rt_files, chunksize=RT_CHUNK_ROWS, top_k=TOP_K):
    """
    Memory-safe pass to find the top_k services by average RPS.
    Writes aggregates to a temp file, then reads it.
    """
    log("Starting PASS 1: Finding Top-K microservices (memory-safe)...")
    if os.path.exists(TEMP_AGG_FILE):
        os.remove(TEMP_AGG_FILE)

    header_written = False
    
    for path in rt_files:
        log(f"  Scanning RT file: {path}")
        if not file_quick_check(path):
            continue
        try:
            for chunk in pd.read_csv(path, chunksize=chunksize, engine="c", on_bad_lines="skip", low_memory=False):
                chunk.columns = chunk.columns.str.strip().str.lower()
                required = {"msname", "metric", "value"}
                if not required.issubset(set(chunk.columns)):
                    continue

                metric_col = chunk["metric"].astype(str)
                mask = metric_col.str.contains("provider", case=False, na=False) | metric_col.str.contains("providerrpc_mcr", case=False, na=False)
                sub = chunk.loc[mask, ["msname", "value"]].copy()
                
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub.dropna(subset=["value"])
                if sub.empty:
                    continue

                # Aggregate sum and count for this chunk
                agg_chunk = sub.groupby("msname")["value"].agg(["sum", "count"]).reset_index()
                
                # Append aggregates to temp file
                agg_chunk.to_csv(TEMP_AGG_FILE, mode='a', header=not header_written, index=False)
                header_written = True

        except Exception as e:
            log(f"  ⚠️ Error reading RT file {path}: {e}")
            continue
    
    if not header_written:
        log("❌ No RT metrics discovered. Aborting.")
        return set()

    # Now, read the *much smaller* aggregate file
    log("  Aggregating temp file to find top-k...")
    df_agg = pd.read_csv(TEMP_AGG_FILE)
    df_final_agg = df_agg.groupby('msname').sum()
    df_final_agg['avg_rps'] = df_final_agg['sum'] / df_final_agg['count']
    
    # Get the top-k service names
    top_ms = df_final_agg.nlargest(top_k, 'avg_rps').index.tolist()
    
    log(f"  Selected top-{len(top_ms)} microservices.")
    os.remove(TEMP_AGG_FILE)
    return set(top_ms)

# ------------------- PASS 2: Build Filtered RT Index -------------------
def build_filtered_rt_index(rt_files, top_k_set, chunksize=RT_CHUNK_ROWS):
    """
    Pass 2: Stream RT files again, but only build an index
    for the services in top_k_set.
    """
    log("Starting PASS 2: Building *filtered* RT index...")
    buckets = defaultdict(list)

    for path in rt_files:
        log(f"  Indexing RT file: {path}")
        if not file_quick_check(path):
            continue
        try:
            for chunk in pd.read_csv(path, chunksize=chunksize, engine="c", on_bad_lines="skip", low_memory=False):
                chunk.columns = chunk.columns.str.strip().str.lower()
                required = {"timestamp", "msname", "metric", "value"}
                if not required.issubset(set(chunk.columns)):
                    continue

                # 1. Filter for provider metrics
                metric_col = chunk["metric"].astype(str)
                mask = metric_col.str.contains("provider", case=False, na=False) | metric_col.str.contains("providerrpc_mcr", case=False, na=False)
                sub = chunk.loc[mask].copy()

                # 2. Filter for ONLY our top-k services
                sub = sub[sub['msname'].isin(top_k_set)]
                if sub.empty:
                    continue
                
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub.dropna(subset=["value"])
                
                # 3. Add to buckets
                for t, ms, v in sub[["timestamp", "msname", "value"]].itertuples(index=False, name=None):
                    try:
                        key = (int(float(t)), str(ms))
                        buckets[key].append(float(v))
                    except Exception:
                        continue
        except Exception as e:
            log(f"  ⚠️ Error reading RT file {path}: {e}")
            continue

    # Collapse buckets to mean rps
    rt_index = {}
    for k, vs in buckets.items():
        try:
            rt_index[k] = float(np.mean(vs))
        except Exception:
            continue
            
    log(f"  Filtered RT index size: {len(rt_index)} entries.")
    if not rt_index:
        log("❌ Filtered RT index is empty. Check your data or TOP_K services. Aborting.")
        sys.exit(1)
        
    return rt_index

# ------------------- RESOURCE ROW STREAMING -------------------
def stream_resource_rows(path):
    """
    Yield dicts for each row in the resource CSV.
    Handles leading index column and normalizes header names to lowercase/stripped.
    """
    if not file_quick_check(path):
        raise RuntimeError(f"Quick-check failed: {path}")
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return
        header = [h.strip().lower() for h in header]
        # drop leading empty/unnamed first column if present
        if not header[0] or header[0].startswith("unnamed") or header[0].isdigit():
            header = header[1:]
        for row in reader:
            # align lengths
            if len(row) == len(header) + 1:
                row = row[1:]
            elif len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]
            yield dict(zip(header, row))

# ------------------- SAMPLE CREATION & AUGMENTATION -------------------
def synthesize_replica_from_rps(rps_value):
    """
    Heuristic to create replica_count from rps.
    Aggressive: 1 pod per 80 RPS.
    """
    try:
        r = float(rps_value)
    except Exception:
        r = 0.0
    # Smarter heuristic: e.g., 1 pod per 80 RPS
    rep = int(math.ceil(r / 80.0))
    rep = max(1, min(rep, 10))
    return rep

def build_obs_from_row(cpu, mem, rps, rep):
    """
    Create 16-dim observation vector as used previously.
    """
    obs = np.zeros(16, dtype=np.float32)
    obs[0] = float(cpu)
    obs[1] = float(mem)
    obs[5] = float(rps) / 500.0 # Normalize by a peak
    obs[9] = float(rep) / 20.0
    obs[10] = obs[9]
    obs[11] = 1.0
    return obs

# ------------------- PASS 3: MAIN DATASET BUILD LOOP (MODIFIED) -------------------
def make_large_dataset(res_files, rt_index, include_msnames):
    """
    Pass 3: Stream resource files, look up in filtered index,
    and generate a high-quality, augmented, balanced dataset.
    """
    log("Starting PASS 3: Dataset creation (Quality-Focused Stream)...")
    rng = np.random.default_rng(RNG_SEED)

    part_id = 0
    dataset_part = []
    total_samples = 0
    action_counts = Counter()

    def save_part():
        nonlocal dataset_part, part_id
        if not dataset_part:
            return
        out_path = os.path.join(OUTPUT_DIR, f"dataset_part_{part_id}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dataset_part, f)
        log(f"  Saved part {part_id}: {len(dataset_part)} samples -> {out_path}")
        dataset_part = []
        part_id += 1

    for res_path in res_files:
        log(f"  Streaming resource file: {res_path}")
        if not file_quick_check(res_path):
            log(f"  Skipping unreadable resource file: {res_path}")
            continue
        try:
            for row_idx, row in enumerate(stream_resource_rows(res_path)):
                msname = (row.get("msname") or "").strip()
                if not msname or msname not in include_msnames:
                    continue

                ts_raw = row.get("timestamp", "")
                try:
                    ts = int(float(ts_raw)) if ts_raw != "" else 0
                except Exception:
                    ts = 0

                # Use the *actual* Alibaba column names, fallback to simple
                cpu_raw = row.get("instance_cpu_usage") or row.get("cpu_utilization") or row.get("cpu") or "0"
                mem_raw = row.get("instance_memory_usage") or row.get("memory_utilization") or row.get("memory") or "0"
                try:
                    cpu = float(cpu_raw)
                    mem = float(mem_raw)
                except Exception:
                    cpu = 0.0
                    mem = 0.0
                
                rps = rt_index.get((int(ts), msname), 0.0)

                # --- 1. FILTER "JUNK" ROWS ---
                if rps < 1.0 and cpu < 0.05:
                    if rng.random() > 0.01: # 1% chance to keep it anyway
                        continue

                rep = synthesize_replica_from_rps(rps)
                action = int(max(0, min(9, rep - 1)))
                obs = build_obs_from_row(cpu, mem, rps, rep)

                # --- 2. STRATEGIC AUGMENTATION ---
                num_copies = 1 # Start with 1 copy (the original)
                
                if rps > MIN_RPS_FOR_AUGMENT:
                    if action == 0:
                        # UNDERSAMPLE the common "1 pod" case
                        num_copies += 1 if rng.random() < 0.2 else 0 # 20% chance of 1 augmentation
                    elif action in [1, 2, 3]:
                        # Slightly oversample
                        num_copies += rng.integers(1, 4) # Add 1-3 copies
                    else:
                        # Aggressively OVERSAMPLE rare, high-stress actions
                        num_copies += rng.integers(5, 15) # Add 5-14 copies

                for _ in range(num_copies):
                    if total_samples >= MAX_SAMPLES_HARD:
                        break
                    
                    if total_samples == 0: # First sample is always original
                        aug_obs = obs
                    else:
                        # Create an augmented copy
                        aug_obs = obs.copy()
                        aug_obs[0] = float(max(0.0, aug_obs[0] * (1.0 + rng.normal(0, 0.03)))) # CPU noise
                        aug_obs[1] = float(max(0.0, aug_obs[1] * (1.0 + rng.normal(0, 0.03)))) # Mem noise
                        aug_obs[5] = float(max(0.0, aug_obs[5] * (1.0 + rng.normal(0, 0.05)))) # RPS noise
                    
                    dataset_part.append((aug_obs, action))
                    total_samples += 1
                    action_counts[action] += 1

                if len(dataset_part) >= PART_SIZE:
                    save_part()

                if total_samples >= MAX_SAMPLES_HARD:
                    log(f"Reached hard sample limit: {MAX_SAMPLES_HARD}")
                    break
            
            if total_samples >= MAX_SAMPLES_HARD:
                break
        
        except Exception as e:
            log(f"  ⚠️ Error streaming resource {res_path} at row ~{row_idx}: {e}")
            continue

        if total_samples >= MAX_SAMPLES_HARD:
            break

    save_part() # flush remaining
    
    log(f"Dataset creation complete: total samples={total_samples}")
    log("Final Action Distribution:")
    for i in range(10):
        log(f"  Action {i} ({(i+1)} pods): {action_counts.get(i, 0)} samples")
        
    return total_samples

# ------------------- MERGE PARTS -------------------
def merge_parts(final_path=FINAL_DATASET):
    log("Merging part files into final pickle...")
    parts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "dataset_part_*.pkl")))
    if not parts:
        log("No parts found. Nothing to merge.")
        return 0
    
    merged = []
    total_samples = 0
    for p in parts:
        try:
            with open(p, "rb") as f:
                batch = pickle.load(f)
            merged.extend(batch)
            log(f"  Loaded {p} ({len(batch)} samples). Total: {len(merged)}")
        except Exception as e:
            log(f"  Failed to load {p}: {e}")

    # Shuffle final dataset
    log(f"Shuffling {len(merged)} total samples...")
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(merged)
    
    log(f"Writing final pickle file: {final_path}")
    with open(final_path, "wb") as f:
        pickle.dump(merged, f)
    
    log(f"Final pickle written: {final_path} ({len(merged)} samples).")
    
    # Cleanup
    log("Cleaning up part files...")
    for p in parts:
        try: os.remove(p)
        except Exception: pass
    try: os.rmdir(OUTPUT_DIR)
    except Exception: pass

    return len(merged)

# ------------------- MAIN -------------------
def main():
    start = time.time()
    res_files = sorted(glob.glob(RESOURCE_GLOB))
    rt_files = sorted(glob.glob(RT_GLOB))

    if not res_files:
        log(f"ERROR: No resource files found matching {RESOURCE_GLOB}")
        sys.exit(1)
    if not rt_files:
        log(f"ERROR: No RT files found matching {RT_GLOB}")
        sys.exit(1)

    log(f"Found {len(res_files)} resource files and {len(rt_files)} RT files.")

    # 1) Build RT index and pick top microservices
    top_ms_set = find_top_k_services_safe(rt_files, chunksize=RT_CHUNK_ROWS, top_k=TOP_K)
    if not top_ms_set:
        log("Top microservice detection failed. Exiting.")
        sys.exit(1)
    
    log(f"Top {len(top_ms_set)} services selected for dataset.")

    # 2) Build FILTERED index for only these top services
    rt_index = build_filtered_rt_index(rt_files, top_ms_set, chunksize=RT_CHUNK_ROWS)

    # 3) Create dataset streaming resource rows and augmenting
    samples = make_large_dataset(res_files, rt_index, top_ms_set)

    # 4) Merge parts into final pickle
    merged_count = merge_parts(FINAL_DATASET)

    elapsed = time.time() - start
    log(f"ALL DONE in {elapsed:.1f}s. Final samples: {merged_count}. Final file: {FINAL_DATASET}")

if __name__ == "__main__":
    main()


