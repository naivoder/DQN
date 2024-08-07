import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


def collect_fixed_states(envs, steps=10):
    states, _ = envs.reset()

    for _ in range(steps):
        actions = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        states, _, _, _, _ = envs.step(actions)

    return states


def clip_reward(reward):
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward


# https://github.com/XinJingHao/PPO-Continuous-Pytorch
def action_adapter(a, max_a):
    return 2 * (a - 0.5) * max_a


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame


def save_sample_state(state):
    frame = preprocess_frame(state)
    plt.imshow(frame, cmap="gray")
    plt.savefig("sample_state.jpg")
    plt.close()


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_metrics(env, metrics):
    episodes = np.array(metrics["episode"])
    run_avg_scores = np.array(metrics["average_score"])
    avg_q_values = np.array(metrics["average_q_value"])

    run_avg_qvals = np.zeros_like(avg_q_values)
    for i in range(len(avg_q_values)):
        run_avg_qvals[i] = np.mean(avg_q_values[max(0, i - 100) : i + 1])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Score", color="tab:blue")
    ax1.plot(episodes, run_avg_scores, label="Average Score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Q Value", color="tab:red")
    ax2.plot(episodes, run_avg_qvals, label="Average Q Value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Average Score vs Average Q Value per Episode in {env}")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_metrics.png")
    plt.close()
