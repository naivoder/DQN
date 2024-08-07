import pandas as pd
from utils import plot_metrics
from argparse import ArgumentParser


def load_and_plot_metrics(env):
    metrics = pd.read_csv(f"csv/{env}_metrics.csv")
    plot_metrics(env, metrics)


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot metrics for a given environment.")
    parser.add_argument("--env", type=str, help="The gym environment name")
    args = parser.parse_args()

    load_and_plot_metrics(args.env)
