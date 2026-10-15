# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

data_1 = os.environ.get(
    "WIRELESS_DATASET_PATH",
    "/kaggle/input/datasets/siddss/real-world-wireless-communication-dataset"
)
data_2 = sorted(
    os.path.join(data_3, data_4)
    for data_3, _, data_5 in os.walk(data_1)
    for data_4 in data_5
    if data_4.endswith(".data")
)

data_6 = sorted(
    os.path.join(data_3, data_4)
    for data_3, _, data_5 in os.walk(data_1)
    for data_4 in data_5
    if data_4.endswith(".json")
)

value_1 = 1024
value_2 = 64
value_3 = 2000
value_4 = 7

data_7 = np.random.default_rng(value_4)

def data_8(data_9):
    data_10 = np.memmap(data_9, dtype=np.complex64, mode="r")
    return data_10

def data_11(data_9, value_5):
    data_10 = data_8(data_9)
    value_6 = len(data_10)

    if value_6 < value_1:
        return []

    value_7 = value_6 - value_1
    data_12 = data_7.integers(0, value_7 + 1, size=value_5)

    data_13 = []

    for value_8 in data_12:
        data_14 = np.asarray(
            data_10[value_8:value_8 + value_1],
            dtype=np.complex64
        ).copy()

        if np.all(np.isfinite(data_14)):
            data_13.append(data_14)

    return data_13

def data_15(data_9):
    data_10 = np.asarray(data_9)

    value_5 = np.mean(data_10)
    value_6 = np.std(data_10)

    if value_6 < 1e-12:
        value_6 = 1.0

    return (data_10 - value_5) / value_6

def data_16(data_9):
    data_10 = data_9.reshape(16, 64)
    data_11 = np.fft.fft(data_10, axis=1)
    data_12 = np.abs(data_11)
    data_13 = np.mean(data_12, axis=0)
    return data_13

def data_17(data_9):
    data_10 = np.abs(data_9)
    data_11 = np.array_split(data_10, 16)

    data_12 = []
    data_13 = []

    for data_14 in data_11:
        data_12.append(np.mean(data_14))
        data_13.append(np.std(data_14))

    return np.concatenate([data_12, data_13])

def data_18(data_9):
    data_10 = np.array_split(data_9, 16)

    data_11 = []
    data_12 = []

    for data_13 in data_10:
        data_11.append(np.mean(data_13))
        data_12.append(np.std(data_13))

    return np.concatenate([data_11, data_12])

def data_19(data_9):
    data_10 = np.abs(data_9) ** 2

    value_5 = np.percentile(data_10, 25)
    data_11 = data_10[data_10 <= value_5]

    if len(data_11) == 0:
        return np.nan

    value_6 = np.mean(data_11)
    value_7 = np.mean(data_10)
    value_8 = max(value_7 - value_6, 1e-12)
    value_6 = max(value_6, 1e-12)

    return 10 * np.log10(value_8 / value_6)

def data_20(data_9):
    value_5 = int(
        data_7.uniform(0.0, 0.2) * len(data_9)
    )

    data_10 = np.roll(data_9, value_5)

    value_6 = data_7.uniform(10.0, 30.0)
    value_7 = np.mean(np.abs(data_10) ** 2)
    value_8 = value_7 / (10 ** (value_6 / 10))

    data_11 = (
        data_7.normal(0, np.sqrt(value_8 / 2), len(data_10))
        +
        1j * data_7.normal(0, np.sqrt(value_8 / 2), len(data_10))
    )

    data_10 = data_10 + data_11

    value_9 = int(
        data_7.uniform(0.0, 0.15) * len(data_10)
    )

    if value_9 > 0:
        value_10 = data_7.integers(
            0,
            len(data_10) - value_9 + 1
        )
        data_10[value_10:value_10 + value_9] = 0

    return data_10

def data_21(data_9):
    data_10 = []

    try:
        with open(data_9, "r") as data_11:
            data_12 = json.load(data_11)
    except:
        return data_10

    def data_13(data_14):
        if isinstance(data_14, dict):
            for data_15, data_16 in data_14.items():
                if "snr" in str(data_15).lower():
                    if isinstance(data_16, (int, float)):
                        data_10.append(float(data_16))
                data_13(data_16)

        elif isinstance(data_14, list):
            for data_15 in data_14:
                data_13(data_15)

    data_13(data_12)

    return data_10

value_5 = len(data_2)
value_6 = value_3 // value_5
value_7 = value_3 % value_5

data_22 = []
data_23 = []

for value_8, data_24 in enumerate(data_2):
    value_9 = value_6 + (1 if value_8 < value_7 else 0)
    data_25 = data_11(data_24, value_9)

    for data_26 in data_25:
        data_22.append(data_26)
        data_23.append(
            os.path.relpath(
                os.path.dirname(data_24),
                data_1
            )
        )

data_22 = np.asarray(data_22)

if len(data_22) > value_3:
    data_22 = data_22[:value_3]
    data_23 = data_23[:value_3]

data_27 = np.asarray([
    data_15(data_24)
    for data_24 in data_22
])

data_28 = np.asarray([
    data_16(data_24)
    for data_24 in data_27
])

data_29 = np.asarray([
    data_17(data_24)
    for data_24 in data_27
])

data_30 = np.asarray([
    data_18(data_24)
    for data_24 in data_28
])

data_31 = np.vstack([
    data_29,
    data_30
])

data_32 = np.concatenate([
    np.zeros(len(data_29), dtype=int),
    np.ones(len(data_30), dtype=int)
])

data_33 = StandardScaler().fit_transform(data_31)

data_34 = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=value_4
).fit_transform(data_33)

figure_1 = plt.figure(figsize=(8, 6))

data_35 = data_32 == 0
data_36 = data_32 == 1

plt.scatter(
    data_34[data_35, 0],
    data_34[data_35, 1],
    s=8,
    alpha=0.6,
    label="Representation 1"
)

plt.scatter(
    data_34[data_36, 0],
    data_34[data_36, 1],
    s=8,
    alpha=0.6,
    label="Representation 2"
)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Figure 1")
plt.legend()
plt.tight_layout()
plt.show()

data_37 = data_27[
    data_7.integers(0, len(data_27))
].copy()

data_38 = data_20(
    data_37.copy()
)

data_39 = stft(
    data_37,
    window="hann",
    nperseg=256,
    noverlap=128,
    return_onesided=False
)

data_40 = stft(
    data_38,
    window="hann",
    nperseg=256,
    noverlap=128,
    return_onesided=False
)

data_41 = np.fft.fftshift(
    np.abs(data_39[2]),
    axes=0
)

data_42 = np.fft.fftshift(
    np.abs(data_40[2]),
    axes=0
)

data_43 = 20 * np.log10(
    data_41 + 1e-12
)

data_44 = 20 * np.log10(
    data_42 + 1e-12
)

data_45 = np.abs(
    data_44 - data_43
)

figure_2 = plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(
    data_43,
    aspect="auto",
    origin="lower"
)
plt.title("Original")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.imshow(
    data_44,
    aspect="auto",
    origin="lower"
)
plt.title("Augmented")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.imshow(
    data_45,
    aspect="auto",
    origin="lower"
)
plt.title("Absolute Difference")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.suptitle("Figure 2")
plt.tight_layout()
plt.show()

data_46 = []

for data_47 in data_6:
    data_46.extend(
        data_21(data_47)
    )

if len(data_46) > 0:
    data_48 = np.asarray(
        data_46,
        dtype=float
    )
    data_49 = "recorded metadata"
else:
    data_48 = np.asarray([
        data_19(data_50)
        for data_50 in data_27
    ])

    data_48 = data_48[
        np.isfinite(data_48)
    ]

    data_49 = "estimated from recordings"

data_50 = data_28.ravel()

data_51 = kurtosis(
    data_28,
    axis=1,
    fisher=False,
    bias=False,
    nan_policy="omit"
)

data_51 = data_51[
    np.isfinite(data_51)
]

data_52 = np.percentile(
    data_48,
    [25, 50, 75]
)

data_53 = [
    [
        "Range",
        f"{np.min(data_48):.3f} to {np.max(data_48):.3f} dB",
        data_49
    ],
    [
        "Percentiles",
        f"{data_52[0]:.3f} / {data_52[1]:.3f} / {data_52[2]:.3f} dB",
        data_49
    ],
    [
        "Amplitude",
        f"{np.mean(data_50):.6f} / {np.std(data_50):.6f} / {np.median(data_50):.6f}",
        "mean / std / median"
    ],
    [
        "Kurtosis",
        f"{np.mean(data_51):.6f} / {np.std(data_51):.6f} / {np.median(data_51):.6f}",
        "mean / std / median"
    ]
]

table_1 = pd.DataFrame(
    data_53,
    columns=[
        "Statistic",
        "Value",
        "Description"
    ]
)

