import scipy.io as sio

data = sio.loadmat(
    "./data/spk/M021519_spk.mat",
    simplify_cells=True
)

unit = data["unit"]

neurons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
#[0, 1, 3, 4, 5, 7, 9, 13, 14, 15, 26, 30, 40]
for i in neurons:
    print("\nNeuron", i)
    print("----------------")
    print("session:", unit[i]["session"])
    print("channel:", unit[i]["ch"])
    print("cluster id:", unit[i]["clust_id"])
    print("area:", unit[i]["area"])
    print("timestamps:", len(unit[i]["timestamps"]))
