import gymnasium as gym

def main():
    env = gym.make("CarRacing-v3", render_mode=None)
    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    env.close()

if __name__ == "__main__":
    main()