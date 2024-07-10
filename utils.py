import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


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


def plot_running_avg(scores, env):
    avg = np.zeros_like(scores)
    for i in range(len(scores)):
        avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(avg)
    plt.title("Running Average per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.savefig(f"metrics/{env}_running_avg.png")
    plt.close()
