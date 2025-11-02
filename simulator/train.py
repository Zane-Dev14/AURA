# FILE: pretrain.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

# 1. Define the Agent's "Brain" (Q-Network)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 2. Create a custom PyTorch Dataset
class AlibabaDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.observations = np.array([d[0] for d in data], dtype=np.float32)
        self.actions = np.array([d[1] for d in data], dtype=np.int64)
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

# --- Main Training Script ---
if __name__ == "__main__":
    
    # --- Hyperparameters ---
    OBS_DIM = 16
    ACTION_DIM = 10
    BATCH_SIZE = 256 # Bigger batch size for faster GPU training
    EPOCHS = 20      # Train for longer on this bigger dataset
    LEARNING_RATE = 1e-4

    # --- Setup GPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    dataset = AlibabaDataset('alibaba_dataset.pkl')
    # num_workers=4 uses 4 extra CPU cores to prep data
    # This feeds the GPU so it never has to wait.
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Loaded {len(dataset)} real-world samples.")

    # --- Initialize Model, Loss, and Optimizer ---
    model = QNetwork(OBS_DIM, ACTION_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- The Training Loop ---
    print("Starting pre-training... This will take 20-60 minutes.")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_obs, batch_actions in train_loader:
            
            # Move data to the GPU
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            
            # 1. Forward pass
            action_logits = model(batch_obs)
            
            # 2. Calculate loss
            loss = criterion(action_logits, batch_actions)
            
            # 3. Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} --- Average Loss: {avg_loss:.6f}")

    # --- Save the Checkpoint ---
    checkpoint_path = "alibaba_clone_agent.pth"
    # This saves the model's weights to a file
    torch.save(model.state_dict(), checkpoint_path)
    
    print(f"\n✅ Pre-training complete!")
    print(f"Agent 'brain' (cloned from Alibaba) saved to {checkpoint_path}")