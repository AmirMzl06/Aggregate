import scipy.io as sio

data = sio.loadmat(
    "./data/spk/M021519_spk.mat",
    simplify_cells=True
)

unit = data["unit"]

neurons = [2,6,11,12,19,20,21,33,34,41,42]
#[0, 1, 3, 4, 5, 7, 9, 13, 14, 15, 26, 30, 40]
for i in neurons:
    print("\nNeuron", i)
    print("----------------")
    print("session:", unit[i]["session"])
    print("channel:", unit[i]["ch"])
    print("cluster id:", unit[i]["clust_id"])
    print("area:", unit[i]["area"])
    print("timestamps:", len(unit[i]["timestamps"]))
