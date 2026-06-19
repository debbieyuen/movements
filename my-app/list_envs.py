import gymnasium as gym

envs = sorted([k for k in gym.registry.keys() if "h1" in k.lower() or "humanoid" in k.lower()])
for e in envs:
    print(e)