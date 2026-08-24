import os
import glob
import scipy.io as sio
import numpy as np

BASE = "/home/mirzaei/sam/result/Aggregate/data/RecogMemory/Data"
SESSION = "P10HMH_092206"
TASK = "NO"
NEURON_DIR = os.path.join(BASE, "sorted", SESSION, TASK)
EVENT_FILE = os.path.join(BASE, "events", SESSION, TASK, "eventsRaw.mat")

def load_neurons(folder):
    neurons = {}
    files = sorted(glob.glob(os.path.join(folder, "*_cells.mat")))
    for f in files:
        name = os.path.basename(f).replace("_cells.mat", "")
        mat = sio.loadmat(f)
        spikes = mat["spikes"][:, 2]
        neurons[name] = spikes
    return neurons

def load_events(path):
    data = sio.loadmat(path)
    return data["events"]

def extract_trials(events):
    trials = []
    STIM_ON = 1
    inds = np.where(events[:, 1] == STIM_ON)[0]
    for i in inds:
        if i + 3 >= len(events):
            continue
        trial = {"stim_on": events[i, 0], "stim_off": events[i + 1, 0], "question_on": events[i + 2, 0], "response": events[i + 3, 0], "exp_id": events[i, 2]}
        trials.append(trial)
    return trials

def align_spikes(neurons, trials):
    aligned = []
    for t, trial in enumerate(trials):
        stim_on = trial["stim_on"]
        response = trial["response"]
        trial_data = {}
        for name, spikes in neurons.items():
            mask = (spikes >= stim_on) & (spikes <= response)
            trial_spikes = spikes[mask]
            trial_spikes = trial_spikes - stim_on
            trial_data[name] = trial_spikes
        aligned.append(trial_data)
    return aligned

print("Loading neurons...")
neurons = load_neurons(NEURON_DIR)
print("Number of neurons:", len(neurons))
print("\nLoading events...")
events = load_events(EVENT_FILE)
print("Events:", events.shape)
print("\nExtracting trials...")
trials = extract_trials(events)
print("Trials:", len(trials))
print("\nAlign spikes...")
trial_spikes = align_spikes(neurons, trials)
print("\nExample Trial 0")
for n, s in trial_spikes[0].items():
    print(n, "spikes:", len(s), s[:10])

# import os
# import glob
# import scipy.io as sio
# import numpy as np

# class RecogMemoryLoader:
#     def __init__(self, data_path, session, task="NO"):
#         self.data_path = data_path
#         self.session = session
#         self.task = task
#         self.neurons = {}
#         self.events = None

#     def load_neurons(self):
#         neuron_path = os.path.join(self.data_path, "sorted", self.session, self.task)
#         neuron_files = sorted(glob.glob(os.path.join(neuron_path, "*_cells.mat")))
#         print("Neuron folder:")
#         print(neuron_path)
#         print("Number of neurons:", len(neuron_files))
#         for file in neuron_files:
#             neuron_name = os.path.basename(file)
#             neuron_name = neuron_name.replace("_cells.mat", "")
#             mat = sio.loadmat(file)
#             spikes = mat["spikes"]
#             spike_times = spikes[:, 2]
#             self.neurons[neuron_name] = spike_times

#     def load_events(self):
#         event_path = os.path.join(self.data_path, "events", self.session, self.task, "eventsRaw.mat")
#         print("\nEvent file:")
#         print(event_path)
#         mat = sio.loadmat(event_path)
#         self.events = mat["events"]

#     def load(self):
#         self.load_neurons()
#         self.load_events()
#         print("\nLoading finished!")
#         print("Neurons:", len(self.neurons))
#         print("Events:", self.events.shape)

# DATA_PATH = "/home/mirzaei/sam/result/Aggregate/data/RecogMemory/Data"
# session = "P10HMH_092206"
# dataset = RecogMemoryLoader(data_path=DATA_PATH, session=session, task="NO")
# dataset.load()

# print("\nExample neurons:")
# for neuron, spikes in list(dataset.neurons.items())[:3]:
#     print(neuron, "number of spikes:", len(spikes))
#     print(spikes[:5])


# import numpy as np

# def extract_trials(events):
#     trials = []
#     STIM_ON = 1
#     stim_indices = np.where(events[:, 1] == STIM_ON)[0]
#     print("Number of stimulus events:", len(stim_indices))
#     for idx in stim_indices:
#         if idx + 3 >= len(events):
#             continue
#         stim_on = events[idx, 0]
#         stim_off = events[idx + 1, 0]
#         question_on = events[idx + 2, 0]
#         response = events[idx + 3, 0]
#         experiment_id = events[idx, 2]
#         trial = {
#             "stim_on": stim_on,
#             "stim_off": stim_off,
#             "question_on": question_on,
#             "response": response,
#             "experiment_id": experiment_id
#         }
#         trials.append(trial)
#     return trials

# trials = extract_trials(dataset.events)
# print("\nTotal trials:", len(trials))
# for i, t in enumerate(trials[:5]):
#     print("\nTrial", i)
#     for k, v in t.items():
#         print(k, v)
# print("\nFirst events:")
# print(dataset.events[:10])


# import numpy as np

# def extract_trial_spikes(neurons, trials):
#     trial_spikes = []
#     for t_idx, trial in enumerate(trials):
#         stim_on = trial["stim_on"]
#         response = trial["response"]
#         one_trial = {}
#         for neuron_name, spikes in neurons.items():
#             mask = (spikes >= stim_on) & (spikes <= response)
#             selected = spikes[mask]
#             relative_time = (selected - stim_on) / 1e6
#             one_trial[neuron_name] = relative_time
#         trial_spikes.append(one_trial)
#     return trial_spikes

# trial_spikes = extract_trial_spikes(dataset.neurons, trials)
# print("\nExample Trial 0")
# for neuron, spikes in trial_spikes[0].items():
#     print(neuron, "n_spikes:", len(spikes), spikes[:10])
