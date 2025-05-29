import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import matplotlib.pyplot as plt
import time
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class ActionCoupledWrapper(Wrapper):
    def __init__(self, env_fn, k: int):
        self.envs = [env_fn() for _ in range(k)]
        self.k = k
        self.action_space = self.envs[0].action_space
        self.observation_space = gym.spaces.Tuple([env.observation_space for env in self.envs])
        self.terminated_envs = [False] * k  # Track terminated state for each environment

    def reset(self, **kwargs):
        self.terminated_envs = [False] * self.k  # Reset the terminated states
        return tuple(env.reset(**kwargs)[0] for env in self.envs), {}

    def step(self, action):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        
        for i, env in enumerate(self.envs):
            if not self.terminated_envs[i]:
                # Only step environments that haven't terminated yet
                o, r, d, t, i_info = env.step(action)
                obs.append(o)
                rewards.append(r)
                dones.append(d)
                truncs.append(t)
                infos.append(i_info)
                
                # Update terminated state for this environment
                if d or t:
                    self.terminated_envs[i] = True
            else:
                # For already terminated environments, return placeholder values
                obs.append(None)  # or the last observation if you prefer
                rewards.append(0)
                dones.append(True)
                truncs.append(True)
                infos.append({})
        
        done_flags = [d or t for d, t in zip(dones, truncs)]
        # Log number of active environments
        active_envs = sum(1 for d in self.terminated_envs if not d)
        logging.info(f"Active environments: {active_envs}")
        
        return tuple(obs), sum(rewards), all(done_flags), False, {"individual_rewards": rewards, "infos": infos}

    def render(self):
        frames = []
        for i, env in enumerate(self.envs):
            if not self.terminated_envs[i]:
                frames.append(env.render())
            else:
                # For terminated environments, use a blank/black frame
                # This assumes the first environment has been rendered at least once
                if frames:
                    h, w, c = frames[0].shape
                    frames.append(np.zeros((h, w, c), dtype=np.uint8))
                else:
                    # If no frames yet, need to get dimensions somehow
                    tmp_obs, _ = env.reset()
                    frame = env.render()
                    env.step(self.action_space.sample())  # Take a dummy step to get back to terminated
                    frames.append(np.zeros_like(frame))
        
        if not frames or any(f is None for f in frames):
            return None

        # Determine grid size (e.g., 2x2 for 4 envs, 3x3 for 9, etc.)
        rows = math.ceil(math.sqrt(self.k))
        cols = math.ceil(self.k / rows)
        h, w, c = frames[0].shape
        grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)

        for idx, frame in enumerate(frames):
            r, c_ = divmod(idx, cols)
            grid[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = frame

        return grid

    def close(self):
        for env in self.envs:
            env.close()

# Main loop
if __name__ == "__main__":
    # env = ActionCoupledWrapper(lambda: gym.make("CartPole-v1", render_mode="rgb_array"), k=4)
    env = ActionCoupledWrapper(lambda: gym.make("MountainCar-v0", render_mode="rgb_array"), k=4)
    obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(10, 10))
    frame = env.render()
    im = ax.imshow(frame)
    plt.axis("off")

    done = False
    total_reward = 0
    step_count = 0
    
    try:
        while not done:
            action = env.action_space.sample()  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            frame = env.render()
            if frame is not None:
                im.set_data(frame)
                plt.draw()
                plt.pause(0.01)
            
            done = terminated or truncated
            
        logging.info(f"Episode finished after {step_count} steps with total reward {total_reward}")
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    finally:
        env.close()
        plt.close()