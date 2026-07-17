import os
import tempfile
from typing import Tuple, Dict

import gdown
import numpy as np

DATA_DIR = "./data""
CACHE_DIR = "./weights_cache/"


def emg_preprocessing(trial_emg, EMG_names):
    """
    Function to clip and normalize each channel of EMG envelopes
    :param trial_emg: list of lists containing EMG envelopes for each successful trial
    :return: list of lists containing the pre-processed EMG envelopes for each successful trial
    """

    # First, keep EMG channels that are good across all recorded sessions
    EMG_names_good = ['EMG_ECRb', 'EMG_ECRl', 'EMG_ECU', 'EMG_EDCr', 'EMG_FCR', 'EMG_FCU', 'EMG_FDP']
    idx_emg = [EMG_names.index(x) for x in EMG_names_good]
    for trial, val_trial in enumerate(trial_emg):
        trial_emg[trial] = val_trial[:, idx_emg]

    trial_emg_np = np.concatenate(trial_emg)

    # EMG clipping
    outlier = np.mean(trial_emg_np, axis=0) + 6 * np.std(trial_emg_np, axis=0)
    for i, val in enumerate(trial_emg):
        for ii in range(len(outlier)):
            trial_emg[i][:, ii] = trial_emg[i][:, ii].clip(max=outlier[ii])

    # EMG normalization
    trial_emg_np_baseline = np.percentile(trial_emg_np, 2, axis=0)
    trial_emg_np_max = np.percentile(trial_emg_np - trial_emg_np_baseline, 90, axis=0)
    for i, val in enumerate(trial_emg):
        trial_emg[i] = (val - trial_emg_np_baseline) / trial_emg_np_max

    return trial_emg, EMG_names_good


class DatasetLoader:
    # Map dataset names to Google Drive URLs
    dataset_names = [
        "Jango_ISO_2015_npz",
        "Mihili_CO_2014_npz",
        "Mihili_RT_2013_2014_npz",
        "erdiff_synthetic_npz",
        "Chewie_CO_2016_npz",
        "Jango_ISO_2015_raw",
        "Mihili_CO_2014_raw",
        "Mihili_RT_2013_2014_raw",
        "Chewie_CO_2016_raw",
    ]
    dataset_original_names = {
        k: k[:-4] for k in dataset_names
    }
    dataset_urls = {
        "Jango_ISO_2015_npz": "https://drive.google.com/file/d/13bfED4LH4j0YhvEpU6yRoSYbFjG8zhJD/view",
        "Mihili_CO_2014_npz": "https://drive.google.com/file/d/19NYpey1xMbiAT7lThevbSJ7t52X6WP6R/view",
        "Mihili_RT_2013_2014_npz": "https://drive.google.com/file/d/1hPEOVkdClP_5Ic1NiO9xbVI8hH4Iz5wm/view",
        "erdiff_synthetic_npz": "https://drive.google.com/file/d/1qfZ0MHEAnqfB8BmRGEGJLkUIWUt3rxJu/view",
        "Chewie_CO_2016_npz": "https://drive.google.com/file/d/1w_bSnycpFmul9yWpJ-kxHTETmevKZcDd/view?usp=sharing",
        "Jango_ISO_2015_raw": "https://drive.google.com/file/d/1TCd3bFniniZdSg5w1bOgsgZ7-0PDA2-a/view?usp=sharing",
        "Mihili_CO_2014_raw": "https://drive.google.com/file/d/10ITSrl_-iuwaPi0KxZBxPnDeCHX0agPz/view?usp=sharing",
        "Mihili_RT_2013_2014_raw": "https://drive.google.com/file/d/1U-IhuqPnL0Bm9rzRwhCgpe4MdRNM1og3/view?usp=sharing",
        "Chewie_CO_2016_raw": "https://drive.google.com/file/d/1J2ukcUyev4n1JkljD61BKZoPXuOa05pP/view?usp=sharing",
    }
    file_types = {
        "Jango_ISO_2015_npz": "zip",
        "Mihili_CO_2014_npz": "zip",
        "Mihili_RT_2013_2014_npz": "zip",
        "erdiff_synthetic_npz": "zip",
        "Chewie_CO_2016_npz": "zip",
        "Jango_ISO_2015_raw": "7z",
        "Mihili_CO_2014_raw": "7z",
        "Mihili_RT_2013_2014_raw": "7z",
        "Chewie_CO_2016_raw": "7z",
    }
    num_days = {
        "Jango_ISO_2015_npz": 20,
        "Mihili_CO_2014_npz": 11,
        "Mihili_RT_2013_2014_npz": 11,
        "erdiff_synthetic_npz": 1,
        "Chewie_CO_2016_npz": 12,
        "Jango_ISO_2015_raw": 20,
        "Mihili_CO_2014_raw": 11,
        "Mihili_RT_2013_2014_raw": 11,
        "Chewie_CO_2016_raw": 12,
    }

    weights_cache: Dict[Tuple[str, int, str, float], Tuple[np.ndarray, np.ndarray]] = {}

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatasetLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_root_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR):
        if hasattr(self, "_initialized"):
            return

        self.data_root_dir = data_root_dir
        self.cache_dir = cache_dir

        if not os.path.exists(self.data_root_dir):
            print(f"Creating {self.data_root_dir} directory...")
            os.makedirs(self.data_root_dir)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._initialized = True

    def _get_cache_filename(self, dataset_name, day_num, trial_start, bin_size):
        """Generate cache filename from parameters"""
        trial_start_str = trial_start if trial_start else "default"
        bin_size_str = f"{bin_size:.4f}".replace('.', 'p')
        return f"{dataset_name}_{day_num}_{trial_start_str}_{bin_size_str}.npz"

    def _get_cache_path(self, dataset_name, day_num, trial_start, bin_size):
        """Get full path to cache file"""
        filename = self._get_cache_filename(dataset_name, day_num, trial_start, bin_size)
        return os.path.join(self.cache_dir, filename)

    def _load_from_disk_cache(self, cache_path):
        """Load data from disk cache if it exists"""
        if os.path.exists(cache_path):
            with np.load(cache_path) as data:
                return data['x'], data['y']
        return None

    def _save_to_disk_cache(self, cache_path, x_np, y_np):
        """Save data to disk cache"""
        np.savez_compressed(cache_path, x=x_np, y=y_np)

    def download_from_drive(self, dataset_name):
        file_type = self.file_types[dataset_name]

        url = self.dataset_urls[dataset_name]
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        compressed_file_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.{file_type}")

        print(f"Downloading {dataset_name} from {download_url} ...")
        gdown.download(download_url, compressed_file_path, quiet=False)

        return compressed_file_path

    def extract_compressed_dataset(self, compressed_file_path, dataset_name):
        file_type = self.file_types[dataset_name]
        dataset_dir = self.get_dataset_folder(dataset_name)

        # Step 3: Unzip into dataset_dir
        print(f"Extracting {compressed_file_path} into {dataset_dir} ...")
        if file_type == "zip":
            import zipfile
            with zipfile.ZipFile(compressed_file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
        elif file_type == "7z":
            import py7zr, shutil
            with py7zr.SevenZipFile(compressed_file_path, mode='r') as archive:
                archive.extractall(path=dataset_dir)
            # files will be saved in dataset_dir/original_name/* -> need to move them to dataset_dir/*
            original_name = self.dataset_original_names[dataset_name]
            original_dir = os.path.join(dataset_dir, original_name)
            for f in os.listdir(original_dir):
                shutil.move(os.path.join(original_dir, f), os.path.join(dataset_dir, f))
            shutil.rmtree(original_dir)

        return dataset_dir

    def prepare_dataset(self, dataset_name: str):
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_dir = self.get_dataset_folder(dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        if os.listdir(dataset_dir):
            return dataset_dir

        # Download the ZIP file using gdown
        compressed_file_path = self.download_from_drive(dataset_name)

        # Unzip into dataset_dir
        dataset_dir = self.extract_compressed_dataset(compressed_file_path, dataset_name)

        # Remove the zip file to save space (optional)
        os.remove(compressed_file_path)
        print(f"Finished! Files are in: {dataset_dir}")
        return dataset_dir

    def get_dataset_folder(self, dataset_name: str):
        return os.path.join(self.data_root_dir, dataset_name)

    def get_dataset_day_filename(self, day_num: int, dataset_name: str):
        dataset_folder = self.get_dataset_folder(dataset_name)
        file_list = os.listdir(dataset_folder)
        file_list.sort()

        if not (0 <= day_num < len(file_list)):
            raise IndexError(f"Index {day_num} out of range [0, {len(file_list)})")

        fname = file_list[day_num]
        return fname

    def get_dataset_day_file_path(self, day_num: int, dataset_name: str):
        if '/' in dataset_name:
            dataset_name = dataset_name.split('/')[-1]

        self.prepare_dataset(dataset_name)

        dataset_folder = self.get_dataset_folder(dataset_name)

        fname = self.get_dataset_day_filename(day_num, dataset_name)
        return os.path.join(dataset_folder, fname)

    def get_raw_dataset_xy(self, dayk_id, dataset_name, trial_start='gocue_time', bin_size=0.05):
        import sys
        sys.path.append('./xds/xds_python/')
        from xds import lab_data

        smooth_size = 0.1  # As mentioned above, we use Gaussian kernels (S.D. = 100 ms) to smooth the binned spikes

        # ============================================= Load day-k data ==================================================================#
        dayk_path = self.get_dataset_day_file_path(day_num=dayk_id, dataset_name=dataset_name)
        dayk_base_path, dayk_file_name = os.path.split(dayk_path)

        dayk_data = lab_data(dayk_base_path, dayk_file_name)
        dayk_data.update_bin_data(bin_size)
        dayk_data.smooth_binned_spikes(bin_size, 'gaussian', smooth_size)
        dayk_unit_names = dayk_data.unit_names

        # -------- Extract smoothed spike counts in trials without temporal alignment --------#
        dayk_spike = dayk_data.get_trials_data_spike_counts('R', trial_start, 0.0, 'end_time', 0)
        # -------- Extract EMG envelops in trials without temporal alignment --------#
        if 'Jango' in dataset_name:
            dayk_label = dayk_data.get_trials_data_EMG('R', trial_start, 0.0, 'end_time', 0)
            dayk_label, _ = emg_preprocessing(dayk_label, dayk_data.EMG_names)
        else:
            dayk_label = dayk_data.get_trials_data_cursor('R', trial_start, 0.0, 'end_time', 0)
            dayk_label = [np.concatenate(arrs, axis=1) for arrs in zip(*dayk_label)]
        # ============================================= Pre-processing ==================================================================#
        dayk_spike = self.spike_zero_padding(dataset_name, dayk_unit_names, dayk_spike)

        return np.vstack(dayk_spike), np.vstack(dayk_label)

    def load_dataset_day(self, day_num: int, dataset_name: str, cache: bool = True, trial_start=None,
                         bin_size=0.05) -> (np.ndarray, np.ndarray):
        """
        Given an integer index day_num, loads the day_num-th file in dataset folder
        (appending '.npz'), and returns x, y as torch.Tensors.
        """
        if trial_start is None:
            if 'Jango' in dataset_name or 'Mihili_RT' in dataset_name:
                trial_start = 'start_time'
            else:
                trial_start = 'gocue_time'

        cache_key = (dataset_name, day_num, trial_start, bin_size)

        # Check in-memory cache first
        if cache and cache_key in self.weights_cache:
            return self.weights_cache[cache_key]

        # Check disk cache
        if cache:
            cache_path = self._get_cache_path(dataset_name, day_num, trial_start, bin_size)
            cached_data = self._load_from_disk_cache(cache_path)
            if cached_data is not None:
                self.weights_cache[cache_key] = cached_data
                return cached_data

        self.prepare_dataset(dataset_name)

        if 'raw' in dataset_name:
            x_np, y_np = self.get_raw_dataset_xy(day_num, dataset_name, trial_start, bin_size)
        else:
            path = self.get_dataset_day_file_path(day_num, dataset_name)

            # load numpy arrays
            with np.load(path) as data:
                x_np = data['x']
                y_np = data['y']

        if cache:
            # Save to disk cache
            cache_path = self._get_cache_path(dataset_name, day_num, trial_start, bin_size)
            self._save_to_disk_cache(cache_path, x_np, y_np)

            # Save to in-memory cache
            self.weights_cache[cache_key] = (x_np, y_np)

        return x_np, y_np

    def get_dataset_num_days(self, dataset_name: str):
        return self.num_days[dataset_name]

    def get_all_unit_names(self, dataset_name: str):
        Jango_names = ['elec93', 'elec92', 'elec94', 'elec95', 'elec75', 'elec96', 'elec85', 'elec97', 'elec86',
                       'elec98', 'elec87', 'elec88', 'elec77', 'elec99', 'elec66', 'elec89', 'elec76', 'elec90',
                       'elec67', 'elec79', 'elec58', 'elec80', 'elec78', 'elec70', 'elec68', 'elec60', 'elec69',
                       'elec50', 'elec59', 'elec40', 'elec49', 'elec100', 'elec83', 'elec84', 'elec73', 'elec74',
                       'elec63', 'elec64', 'elec53', 'elec54', 'elec43', 'elec55', 'elec44', 'elec45', 'elec33',
                       'elec46', 'elec34', 'elec65', 'elec24', 'elec56', 'elec35', 'elec47', 'elec25', 'elec57',
                       'elec26', 'elec36', 'elec27', 'elec37', 'elec28', 'elec38', 'elec29', 'elec48', 'elec19',
                       'elec39', 'elec81', 'elec82', 'elec71', 'elec72', 'elec61', 'elec62', 'elec51', 'elec52',
                       'elec41', 'elec42', 'elec31', 'elec32', 'elec21', 'elec22', 'elec11', 'elec12', 'elec2',
                       'elec23', 'elec3', 'elec13', 'elec4', 'elec14', 'elec15', 'elec5', 'elec16', 'elec6', 'elec17',
                       'elec7', 'elec8', 'elec18', 'elec20', 'elec9']
        other_names = ['elec78', 'elec88', 'elec68', 'elec58', 'elec56', 'elec48', 'elec57', 'elec38', 'elec47',
                       'elec28', 'elec37', 'elec27', 'elec36', 'elec18', 'elec45', 'elec17', 'elec46', 'elec8',
                       'elec35', 'elec16', 'elec24', 'elec7', 'elec26', 'elec6', 'elec25', 'elec5', 'elec15', 'elec4',
                       'elec14', 'elec3', 'elec13', 'elec2', 'elec77', 'elec67', 'elec76', 'elec66', 'elec75', 'elec65',
                       'elec74', 'elec64', 'elec73', 'elec54', 'elec63', 'elec53', 'elec72', 'elec43', 'elec62',
                       'elec55', 'elec61', 'elec44', 'elec52', 'elec33', 'elec51', 'elec34', 'elec41', 'elec42',
                       'elec31', 'elec32', 'elec21', 'elec22', 'elec11', 'elec23', 'elec10', 'elec12', 'elec96',
                       'elec87', 'elec95', 'elec86', 'elec94', 'elec85', 'elec93', 'elec84', 'elec92', 'elec83',
                       'elec91', 'elec82', 'elec90', 'elec81', 'elec89', 'elec80', 'elec79', 'elec71', 'elec69',
                       'elec70', 'elec59', 'elec60', 'elec50', 'elec49', 'elec40', 'elec39', 'elec30', 'elec29',
                       'elec19', 'elec20', 'elec1', 'elec9']
        if dataset_name.startswith('Jango'):
            return Jango_names
        else:
            return other_names

    def spike_zero_padding(self, dataset_name, spike_unit_names, spike):
        max_unit_names = self.get_all_unit_names(dataset_name)
        N_unit = len(max_unit_names)

        idx = [list(max_unit_names).index(e) for e in spike_unit_names]
        spike_ = [np.zeros((s.shape[0], N_unit)) for s in spike]
        for k in range(len(spike)):
            spike_[k][:, idx] = spike[k]
        return spike_
