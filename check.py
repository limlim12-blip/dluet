import numpy as np
import pandas as pd

path = "/home/lim/Projects/dluet/saved/models/limlim/1101_062529/val_stats_epoch81.npz"

cm = np.load(path)["cm_flow"].astype(float)
np.fill_diagonal(cm, 0)
pairs = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:20], cm.shape))[0]
print(path)
for t, p in pairs:
    print(t, "->", p, int(cm[t, p]))


path = "/home/lim/Projects/dluet/saved/models/limlim/1030_063547/val_stats_epoch14.npz"

cm = np.load(path)["cm_flow"].astype(float)
np.fill_diagonal(cm, 0)
pairs = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:20], cm.shape))[0]
print(path)
for t, p in pairs:
    print(t, "->", p, int(cm[t, p]))

df = pd.read_csv("./data/cesnet-quic22/dataset-statistics/app.csv")
a = {label: idx for idx, label in enumerate(sorted(df["LABEL"].to_list()))}
print(a)
