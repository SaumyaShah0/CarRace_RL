import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


def make_env_eval(render_mode="human"):
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=True,
        )
        env = Monitor(env)
        return env
    return _init


def main():
    model_path = "car_race_env/logs_car_race/best_model/best_model.zip"

    # Load env for rendering
    env = DummyVecEnv([make_env_eval("human")])
    env = VecTransposeImage(env)

    model = PPO.load(model_path, env=env)

    for ep in range(5):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = done_arr[0]
            total_reward += reward[0]
            time.sleep(1 / 30)

        print(f"Episode {ep + 1} reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()