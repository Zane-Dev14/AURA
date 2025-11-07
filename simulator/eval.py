import torch, numpy as np
from boutique_env import K8sAutoscaleEnv
from train import QNetwork, OBS_DIM, ACTION_DIM, SAVE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = K8sAutoscaleEnv("config.yaml")
agent = "api"

net = QNetwork(OBS_DIM, ACTION_DIM).to(device)
net.load_state_dict(torch.load(f"{SAVE_DIR}/{agent}_best.pth", map_location=device))
net.eval()

obs_dict, _ = env.reset()
obs = np.array(obs_dict[agent], dtype=np.float32)

done = False
total_reward = 0
while not done:
    with torch.no_grad():
        q, _ = net(obs)
    act = int(q.argmax().item())
    obs_dict, rewards, terms, truncs, infos = env.step({agent: act})
    total_reward += rewards[agent]
    done = terms[agent] or truncs[agent]
    obs = np.array(obs_dict[agent], dtype=np.float32)
print(f"Total reward: {total_reward:.2f}")
