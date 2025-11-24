import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


# -------------------------
# ENV FACTORY
# -------------------------
def make_env():
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode=None,
            continuous=True,
        )
        env = Monitor(env)
        return env
    return _init


# -------------------------
# MAIN TRAINING
# -------------------------
def main():
    LOG_DIR = "logs_car_race"
    CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
    BEST_DIR = os.path.join(LOG_DIR, "best_model")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    # Vectorized env (SB3 requires VecEnv)
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecTransposeImage(eval_env)

    # Save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_car_racing",
    )

    # Evaluate periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=BEST_DIR,
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    # -------------------------
    # PPO MODEL (GPU ENABLED)
    # -------------------------
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cuda",    #  <<<<  GPU HERE
    )

    # -------------------------
    # TRAIN
    # -------------------------
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(os.path.join(LOG_DIR, "ppo_car_racing_final"))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()