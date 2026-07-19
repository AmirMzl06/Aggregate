from utils.dataset_loader import DatasetLoader

loader = DatasetLoader()

datasets = [
    "Jango_ISO_2015_npz",
    "Mihili_CO_2014_npz",
    "Mihili_RT_2013_2014_npz",
    "Chewie_CO_2016_npz",
]

for ds in datasets:
    print("=" * 60)
    print(f"Preparing {ds} ...")
    path = loader.prepare_dataset(ds)
    print(f"Done -> {path}")

print("\nAll monkey datasets downloaded successfully.")
