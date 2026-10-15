# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import os

data_1 = os.environ.get(
    "WIRELESS_DATASET_PATH",
    "/kaggle/input/datasets/siddss/real-world-wireless-communication-dataset"
)

print(f"Root: {data_1}")
print("=" * 100)

for path_1, dirs_1, files_1 in os.walk(data_1):
    level_1 = path_1.replace(data_1, "").count(os.sep)
    indent_1 = "    " * level_1
    print(f"{indent_1}{os.path.basename(path_1)}/")

    indent_2 = "    " * (level_1 + 1)

    for file_1 in sorted(files_1):
        path_2 = os.path.join(path_1, file_1)

        try:
            size_1 = os.path.getsize(path_2)
        except:
            size_1 = None

        print(f"{indent_2}{file_1} | {size_1} bytes")

print("=" * 100)

dirs_2 = []
files_2 = []

for path_1, dirs_1, files_1 in os.walk(data_1):
    for dir_1 in dirs_1:
        dirs_2.append(os.path.join(path_1, dir_1))

    for file_1 in files_1:
        files_2.append(os.path.join(path_1, file_1))

print("Total directories:", len(dirs_2))
print("Total files:", len(files_2))

print("\nAll directories:")
for path_1 in sorted(dirs_2):
    print(path_1)

print("\nAll files:")
for path_1 in sorted(files_2):
    print(path_1)