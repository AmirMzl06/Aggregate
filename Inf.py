import scipy.io as sio

data = sio.loadmat(
    "./data/spk/M021519_spk.mat",
    simplify_cells=True
)

unit = data["unit"]

print("Number of neurons:", len(unit))

for i in range(min(5, len(unit))):
    print("\nNeuron", i)
    print("----------------")
    print("session:", unit[i]["session"])
    print("channel:", unit[i]["ch"])
    print("cluster id:", unit[i]["clust_id"])
    print("area:", unit[i]["area"])
    print("timestamps:", len(unit[i]["timestamps"]))
