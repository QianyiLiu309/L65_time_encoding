import matplotlib.pyplot as plt
import numpy as np
import argparse
import random


def plot(all_times, all_probs, title="", label=""):
    plt.title(title)
    plt.plot(all_times, all_probs, label=label, alpha=0.7, linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Predicted link probability")
    plt.ylim(-0.01, 1.01)


def plot_real(real_times):
    # plt.scatter(x=real_times, y=[1]*len(real_times), marker='x')
    for etime in real_times:
        plt.axvline(x=etime, color="red", ls="--", linewidth=1, alpha=0.8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=None)
    parser.add_argument("--dst", type=int, default=None)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--title", default="")

    args = parser.parse_args()

    sd = args.src, args.dst

    plt.rcParams["figure.figsize"] = (12, 6)
    
    shared_keys = None
    
    if sd[0] is None:
        for f in args.files:
            d = set(np.load(f, allow_pickle=True).tolist())
            shared_keys = d if shared_keys is None else shared_keys.intersection(d)
        sd = random.choice(list(shared_keys))
        
    for f in args.files:
        d = np.load(f, allow_pickle=True).tolist()

        times, probs, real_times = d[sd]
        name = f.split("/")[-1].split(".npy")[0]
        plot(times, probs, title=args.title + f" src={sd[0]}, dst={sd[1]}", label=name)

    plot_real(real_times)

    plt.legend()
    plt.show()
