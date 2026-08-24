import os
import glob
import scipy.io as sio
import numpy as np

class RecogMemoryLoader:
    def __init__(self, data_path, session, task="NO"):
        self.data_path = data_path
        self.session = session
        self.task = task
        self.neurons = {}
        self.events = None

    def load_neurons(self):
        neuron_path = os.path.join(self.data_path, "sorted", self.session, self.task)
        neuron_files = sorted(glob.glob(os.path.join(neuron_path, "*_cells.mat")))
        print("Neuron folder:")
        print(neuron_path)
        print("Number of neurons:", len(neuron_files))
        for file in neuron_files:
            neuron_name = os.path.basename(file)
            neuron_name = neuron_name.replace("_cells.mat", "")
            mat = sio.loadmat(file)
            spikes = mat["spikes"]
            spike_times = spikes[:, 2]
            self.neurons[neuron_name] = spike_times

    def load_events(self):
        event_path = os.path.join(self.data_path, "events", self.session, self.task, "eventsRaw.mat")
        print("\nEvent file:")
        print(event_path)
        mat = sio.loadmat(event_path)
        self.events = mat["events"]

    def load(self):
        self.load_neurons()
        self.load_events()
        print("\nLoading finished!")
        print("Neurons:", len(self.neurons))
        print("Events:", self.events.shape)

DATA_PATH = "/home/mirzaei/sam/result/Aggregate/data/RecogMemory/Data"
session = "P10HMH_092206"
dataset = RecogMemoryLoader(data_path=DATA_PATH, session=session, task="NO")
dataset.load()

print("\nExample neurons:")
for neuron, spikes in list(dataset.neurons.items())[:3]:
    print(neuron, "number of spikes:", len(spikes))
    print(spikes[:5])

print("\nFirst events:")
print(dataset.events[:10])
