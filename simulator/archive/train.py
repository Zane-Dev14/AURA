#!/usr/bin/env python3
"""
train_v4_GPU_masking.py

This version fixes the major CPU bottleneck from v3.

--- FASTER UPGRADES ---
- CRITICAL FIX: Removed obs.copy() from __getitem__.
- Masking is now performed on the GPU on the entire batch at once.
- Removed unnecessary time.sleep() (GPU cooldown).
- Increased BATCH_SIZE to 8192 for better GPU utilization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. IMPROVED Q-Network (Same as before)
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Custom Dataset (FAST VERSION)
# ============================================================

class AlibabaDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # unpack structure: [(obs,action), (obs,action), ...]
        self.observations = np.array([d[0] for d in data], dtype=np.float32)
        self.actions      = np.array([d[1] for d in data], dtype=np.int64)
        print("✅ Dataset loaded into RAM.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # --- FIX ---
        # Return the raw data directly. This is extremely fast.
        # Masking will be done on the GPU in the training loop.
        return self.observations[idx], self.actions[idx]


# ============================================================
# 3. Warm Start Loader (Same as before)
# ============================================================

def warm_start_load(new_model, old_checkpoint_path):
    try:
        old_state = torch.load(old_checkpoint_path, map_location="cpu")

        new_state = new_model.state_dict()
        matched = 0
        skipped = 0

        for k in new_state.keys():
            if k in old_state and old_state[k].shape == new_state[k].shape:
                new_state[k] = old_state[k]
                matched += 1
            else:
                skipped += 1

        new_model.load_state_dict(new_state)
        print(f"✅ Warm-start loaded. Matched: {matched} layers | Skipped: {skipped} layers")
    except Exception as e:
        print(f"⚠️ Warm-start failed: {e}")


# ============================================================
# 4. Weighted Loss Helper (Same as before)
# ============================================================

def compute_class_weights(dataset, num_classes=10):
    counts = np.zeros(num_classes, dtype=np.int64)
    for a in dataset.actions:
        counts[a] += 1
    
    # Handle classes with 0 samples to avoid division by zero
    counts_safe = np.maximum(counts, 1)
    weights = 1.0 / counts_safe
    
    # Set weight to 0 for classes that never appear
    weights[counts == 0] = 0
    
    # Normalize
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # CONFIG
    # -----------------------------
    DATASET_PATH   = "alibaba_dataset_large.pkl"
    CHECKPOINT_OLD = "alibaba_clone_agent_v2.pth"   # warm start
    CHECKPOINT_NEW = "alibaba_clone_agent_v4.pth"   # <-- New output file

    OBS_DIM    = 16
    ACTION_DIM = 10

    # --- FASTER CONFIG ---
    BATCH_SIZE = 8192       # Even larger batch for better GPU saturation
    EPOCHS     = 120        # Max epochs
    EARLY_STOP_PATIENCE = 15  # Stop if no improvement after 15 epochs
    # ---
    
    LR         = 1e-4
    VALIDATION_SPLIT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load Dataset
    # -----------------------------
    dataset = AlibabaDataset(DATASET_PATH)
    class_weights = compute_class_weights(dataset).to(device)

    val_size   = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- FASTER DATALOADERS ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Speeds up CPU-to-GPU transfer
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True  # Speeds up CPU-to-GPU transfer
    )
    # ---

    # -----------------------------
    # Build Model
    # -----------------------------
    model = QNetwork(OBS_DIM, ACTION_DIM).to(device)

    # Warm start
    warm_start_load(model, CHECKPOINT_OLD)

    # Loss & Optimizer
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    print("🔥 Starting upgraded training...")

    best_val_accuracy = 0.0
    epochs_no_improve = 0  # Counter for early stopping
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            # --- FIX: GPU-SIDE MASKING ---
            # This is done on the entire batch at once, very fast.
            
            # Cap RPS at 0.4
            batch_obs[:, 5] = torch.clamp(batch_obs[:, 5], max=0.4)
            # Zero out replica features
            batch_obs[:, 9] = 0.0
            batch_obs[:, 10] = 0.0
            batch_obs[:, 11] = 0.0
            # ---

            optimizer.zero_grad()

            # ---- MIXED PRECISION ----
            with torch.cuda.amp.autocast():
                logits = model(batch_obs)
                loss   = criterion(logits, batch_actions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        total_val_loss = 0
        correct = 0
        total   = 0

        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)

                # --- FIX: GPU-SIDE MASKING (MUST BE IDENTICAL TO TRAIN) ---
                batch_obs[:, 5] = torch.clamp(batch_obs[:, 5], max=0.4)
                batch_obs[:, 9] = 0.0
                batch_obs[:, 10] = 0.0
                batch_obs[:, 11] = 0.0
                # ---

                with torch.cuda.amp.autocast():
                    logits = model(batch_obs)
                    loss   = criterion(logits, batch_actions)

                total_val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total   += batch_actions.size(0)
                correct += (predicted == batch_actions).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{EPOCHS} [{epoch_time:.2f}s] --- "
              f"Train Loss: {avg_train_loss:.5f} | "
              f"Val Loss: {avg_val_loss:.5f} | "
              f"Val Acc: {val_accuracy:.2f}%")

        # --- EARLY STOPPING LOGIC ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), CHECKPOINT_NEW)
            print(f"  ✅ New best model saved! Acc = {val_accuracy:.2f}%")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1 # Increment counter
            print(f"  (No improvement for {epochs_no_improve} epoch(s))")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"⚠️  Early stopping! No improvement in {EARLY_STOP_PATIENCE} epochs.")
            break
        # ---

        # --- FIX: REMOVED UNNECESSARY GPU COOLDOWN ---
        # if epoch % 30 == 0:
        #     print("🧊 Cooling GPU for 20 seconds...")
        #     time.sleep(20)

    total_training_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_training_time / 60:.2f} minutes!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Saved model: {CHECKPOINT_NEW}")