import re
from matplotlib import pyplot as plt

if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to training logs")
    args = ap.parse_args()

    with open(args.logfile, 'r') as f:

        logs = list([x.strip() for x in f.readlines() if "Epoch" in x])
        train_logs = [{} for _ in range(len(logs))]
        
        for idx, log in enumerate(logs):
            m = re.match("Epoch: (\d+), Train loss: (.*), Val loss: (.*), *", log)
            train_logs[idx]["epoch"] = int(m.match(1))
            train_logs[idx]["train_loss"] = float(m.match(2))
            train_logs[idx]["val_loss"] = float(m.match(3))

    plt.plot([x["epoch"] for x in train_logs], [x["train_loss"] for x in train_logs], label="training loss")
    plt.plot([x["epoch"] for x in train_logs], [x["val_loss"] for x in train_logs], label="val loss")
    plt.legend()

    plt.savefig("utils/logs.png")
