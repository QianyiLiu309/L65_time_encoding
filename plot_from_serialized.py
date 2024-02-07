import matplotlib.pyplot as plt
import numpy as np
import argparse
import random


def plot(all_times, all_probs, title="", label=""):
    plt.title(title)
    plt.plot(all_times, all_probs, label=label)
    plt.xlabel("Time")
    plt.ylabel("Predicted link probability")
    plt.ylim(-0.01, 1.01)


def plot_real(real_times):
    # plt.scatter(x=real_times, y=[1]*len(real_times), marker='x')
    for etime in real_times:
        plt.axvline(x=etime, color="red", ls="--", linewidth=6, alpha=0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=None)
    parser.add_argument("--dst", type=int, default=None)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--title", default="")

    args = parser.parse_args()

    sd = args.src, args.dst

    plt.rcParams["figure.figsize"] = (12, 6)

    for f in args.files:
        d = np.load(f, allow_pickle=True).tolist()
        if sd[0] is None:
            sd = random.choice(list(d.keys()))

        times, probs, real_times = d[sd]
        plot(times, probs, title=args.title + f" src={sd[0]}, dst={sd[1]}")

    plot_real(real_times)

    plt.show()
