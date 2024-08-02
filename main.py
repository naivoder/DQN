import gymnasium as gym
import utils
from dqn import DQNAgent
import numpy as np
import os
import warnings
from argparse import ArgumentParser
import pandas as pd
from preprocess import AtariEnv
from ale_py import ALEInterface, LoggerMode
from config import environments
import torch 

warnings.simplefilter("ignore")
ALEInterface.setLoggerMode(LoggerMode.Error)


def run_dqn(args):

    def make_env():
        # need to pass this func to constructor without calling()
        return AtariEnv(
            args.env,
            shape=(84, 84),
            repeat=4,
            clip_rewards=True,
        ).make()

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(args.n_envs)])
    save_prefix = args.env.split("/")[-1]

    print(f"\nEnvironment: {save_prefix}")
    print(f"Obs.Space: {envs.single_observation_space.shape}")
    print(f"Act.Space: {envs.single_action_space.n}")

    agent = DQNAgent(
        args.env,
        envs.single_observation_space.shape,
        envs.single_action_space.n,
        mem_size=100000,
        batch_size=64,
        eps_dec=1e-6,
        replace_target_count=1000,
    )

    if args.continue_training:
        if os.path.exists(f"weights/{save_prefix}_dqn.pt"):
            agent.load_checkpoint()

    best_score = -np.inf
    avg_score = np.nan
    score = np.zeros(args.n_envs)
    history, metrics = [], []

    states, _ = envs.reset()
    for i in range(args.n_steps):
        actions = [agent.choose_action(state) for state in states]

        next_states, rewards, term, trunc, _ = envs.step(actions)

        for j in range(args.n_envs):
            agent.store_transition(
                states[j],
                actions[j],
                rewards[j],
                next_states[j],
                term[j] or trunc[j],
            )

            score[j] += rewards[j]
            if term[j] or trunc[j]:
                history.append(score[j])
                score[j] = 0

            agent.learn()
            agent.decrement_epsilon()
            # agent.q.scheduler.step()
        states = next_states

        if len(history) > 0:
            avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoint()

        metrics.append(
            {
                "episode": i + 1,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )

        ep_str = f"[Epoch {i + 1:05}/{args.n_steps}]"
        g_str = f"  Completed Games = {len(history)}"
        avg_str = f"  Average Score = {avg_score:.2f}"
        eps_str = f"  Epsilon = {agent.epsilon:.4f}"
        print(ep_str + g_str + avg_str + eps_str, end="\r")

    torch.save(agent.q.state_dict(), f"weights/{save_prefix}_q_final.pt")
    save_results(args.env, history, metrics, agent)


def save_results(env_name, history, metrics, agent):
    save_prefix = env_name.split("/")[-1]
    utils.plot_running_avg(history, save_prefix)
    df = pd.DataFrame(metrics)
    df.to_csv(f"metrics/{save_prefix}_metrics.csv", index=False)
    save_best_version(env_name, agent)


def save_best_version(env_name, agent, seeds=10):
    # actually, its almost always better to use the final weights...
    # agent.load_checkpoint()

    best_total_reward = float("-inf")
    best_frames = None

    env = AtariEnv(
        env_name,
        shape=(84, 84),
        repeat=4,
        clip_rewards=False,
        no_ops=0,
        fire_first=False,
    ).make()

    save_prefix = env_name.split("/")[-1]

    for _ in range(seeds):
        state, _ = env.reset()

        frames = []
        total_reward = 0

        term, trunc = False, False
        while not term and not trunc:
            frames.append(env.render())

            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            total_reward += reward
            state = next_state

        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_frames = frames

    save_prefix = env_name.split("/")[-1]
    utils.save_animation(best_frames, f"environments/{save_prefix}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    parser.add_argument(
        "--n_steps",
        default=100000,
        type=int,
        help="Number of learning steps to run during training",
    )
    parser.add_argument(
        "--n_envs",
        default=32,
        type=int,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--continue_training",
        default=True,
        type=bool,
        help="Continue training from saved weights.",
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        run_dqn(args)
    else:
        for env_name in environments:
            args.env = env_name
            run_dqn(args)
