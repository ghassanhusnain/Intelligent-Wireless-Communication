# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import os
import math
import hashlib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE

data_1 = os.environ.get(
    "WIRELESS_DATASET_PATH",
    "/kaggle/input/datasets/siddss/real-world-wireless-communication-dataset"
)

value_1 = 2000
value_2 = 1024
value_3 = 64
value_4 = 128
value_5 = 3
value_6 = 4
value_7 = 111
value_8 = 0.15
value_9 = 0.07
value_10 = 1e-4
value_11 = 1e-4
value_12 = 4

random.seed(value_7)
np.random.seed(value_7)
torch.manual_seed(value_7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(value_7)

device_1 = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data_2 = sorted(
    os.path.join(data_3, data_4)
    for data_3, _, data_5 in os.walk(data_1)
    for data_4 in data_5
    if data_4.endswith(".data")
)

data_6 = [
    os.path.relpath(
        os.path.dirname(data_3),
        data_1
    )
    for data_3 in data_2
]

data_7 = {
    data_8: value_13
    for value_13, data_8 in enumerate(sorted(set(data_6)))
}

value_14 = value_1 // len(data_2)
value_15 = value_1 % len(data_2)

data_8 = np.random.default_rng(value_7)

data_9 = []
data_10 = []

for value_16, data_11 in enumerate(data_2):
    value_17 = value_14 + (1 if value_16 < value_15 else 0)

    data_12 = np.memmap(
        data_11,
        dtype=np.float32,
        mode="r"
    )

    data_12 = data_12.reshape(-1, 2)

    value_18 = len(data_12) - value_2

    data_13 = data_8.integers(
        0,
        value_18 + 1,
        size=value_17
    )

    for value_19 in data_13:
        data_14 = np.asarray(
            data_12[
                value_19:value_19 + value_2
            ],
            dtype=np.float32
        ).copy()

        data_15 = np.mean(
            data_14,
            axis=0,
            keepdims=True
        )

        data_16 = np.std(
            data_14,
            axis=0,
            keepdims=True
        )

        data_16[data_16 < 1e-7] = 1.0

        data_14 = (
            data_14 - data_15
        ) / data_16

        if np.all(np.isfinite(data_14)):
            data_9.append(data_14)
            data_10.append(
                data_7[data_6[value_16]]
            )

data_9 = np.asarray(
    data_9,
    dtype=np.float32
)

data_10 = np.asarray(
    data_10,
    dtype=np.int64
)

if len(data_9) > value_1:
    data_9 = data_9[:value_1]
    data_10 = data_10[:value_1]

data_11 = np.random.default_rng(
    value_7
).permutation(len(data_9))

data_12 = data_11[:1608]
data_13 = data_11[1608:1808]
data_14 = data_11[1808:2000]

data_15 = torch.tensor(
    data_9,
    dtype=torch.float32
)

data_16 = TensorDataset(
    data_15[data_12]
)

data_17 = TensorDataset(
    data_15[data_13]
)

def data_18(data_19, value_20=1.0):
    data_20 = data_19.clone()

    if value_20 <= 0:
        return data_20

    data_21 = torch.complex(
        data_20[..., 0],
        data_20[..., 1]
    )

    value_21 = int(
        value_20 * 0.2 * value_2
    )

    if value_21 > 0:
        for value_22 in range(
            data_21.shape[0]
        ):
            value_23 = int(
                torch.randint(
                    -value_21,
                    value_21 + 1,
                    (1,),
                    device=data_21.device
                ).item()
            )

            data_21[value_22] = torch.roll(
                data_21[value_22],
                value_23,
                dims=0
            )

    data_22 = (
        10
        +
        20
        *
        torch.rand(
            data_21.shape[0],
            device=data_21.device
        )
    )

    data_22 = (
        30
        -
        value_20
        *
        (30 - data_22)
    )

    data_23 = torch.mean(
        torch.abs(data_21) ** 2,
        dim=1
    )

    data_24 = data_23 / (
        10 ** (data_22 / 10)
    )

    data_25 = (
        torch.randn_like(
            data_21.real
        )
        +
        1j
        *
        torch.randn_like(
            data_21.real
        )
    )

    data_25 = data_25 * torch.sqrt(
        data_24[:, None] / 2
    )

    data_21 = data_21 + data_25

    value_24 = int(
        value_20 * 0.15 * value_2
    )

    if value_24 > 0:
        for value_25 in range(
            data_21.shape[0]
        ):
            value_26 = int(
                torch.randint(
                    0,
                    value_24 + 1,
                    (1,),
                    device=data_21.device
                ).item()
            )

            if value_26 > 0:
                value_27 = int(
                    torch.randint(
                        0,
                        value_2 - value_26 + 1,
                        (1,),
                        device=data_21.device
                    ).item()
                )

                data_21[
                    value_25,
                    value_27:value_27 + value_26
                ] = 0

    data_20 = torch.stack(
        [
            data_21.real,
            data_21.imag
        ],
        dim=-1
    )

    return data_20.float()

class data_19(nn.Module):
    def __init__(self):
        super().__init__()

        self.data_1 = nn.MultiheadAttention(
            value_4,
            value_6,
            dropout=0.1,
            batch_first=True
        )

        self.data_2 = nn.Sequential(
            nn.Linear(
                value_4,
                256
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                256,
                value_4
            ),
            nn.Dropout(0.1)
        )

        self.data_3 = nn.LayerNorm(
            value_4
        )

        self.data_4 = nn.LayerNorm(
            value_4
        )

    def forward(self, data_1):
        data_2, data_3 = self.data_1(
            data_1,
            data_1,
            data_1,
            need_weights=True,
            average_attn_weights=False
        )

        data_4 = self.data_3(
            data_1 + data_2
        )

        data_5 = self.data_2(
            data_4
        )

        data_6 = self.data_4(
            data_4 + data_5
        )

        return data_6, data_3

class data_20(nn.Module):
    def __init__(self):
        super().__init__()

        self.data_1 = nn.Linear(
            value_3 * 2,
            value_4
        )

        self.data_2 = nn.Parameter(
            torch.randn(
                1,
                value_2 // value_3,
                value_4
            ) * 0.02
        )

        self.data_3 = nn.Parameter(
            torch.zeros(
                1,
                1,
                value_4
            )
        )

        self.data_4 = nn.ModuleList([
            data_19()
            for _ in range(value_5)
        ])

        self.data_5 = nn.Sequential(
            nn.Linear(
                value_4,
                value_4
            ),
            nn.GELU(),
            nn.Linear(
                value_4,
                value_4
            )
        )

        self.data_6 = nn.Linear(
            value_4,
            value_3 * 2
        )

    def data_7(self, data_8):
        data_9 = data_8.reshape(
            data_8.shape[0],
            value_2 // value_3,
            value_3,
            2
        )

        data_9 = data_9.reshape(
            data_8.shape[0],
            value_2 // value_3,
            value_3 * 2
        )

        return data_9

    def forward(
        self,
        data_8,
        value_28=0.0
    ):
        data_9 = self.data_7(
            data_8
        )

        data_10 = self.data_1(
            data_9
        )

        data_11 = torch.zeros(
            data_10.shape[:2],
            dtype=torch.bool,
            device=data_10.device
        )

        if value_28 > 0:
            data_11 = (
                torch.rand(
                    data_10.shape[:2],
                    device=data_10.device
                )
                <
                value_28
            )

            data_12 = ~data_11.any(
                dim=1
            )

            if data_12.any():
                data_13 = torch.where(
                    data_12
                )[0]

                data_14 = torch.randint(
                    0,
                    data_10.shape[1],
                    (len(data_13),),
                    device=data_10.device
                )

                data_11[
                    data_13,
                    data_14
                ] = True

            data_10 = torch.where(
                data_11.unsqueeze(-1),
                self.data_3.expand(
                    data_10.shape[0],
                    data_10.shape[1],
                    -1
                ),
                data_10
            )

        data_10 = (
            data_10
            +
            self.data_2
        )

        data_15 = []

        for data_16 in self.data_4:
            data_10, data_17 = data_16(
                data_10
            )

            data_15.append(
                data_17
            )

        data_18 = self.data_6(
            data_10
        )

        data_19 = torch.mean(
            data_10,
            dim=1
        )

        data_20 = F.normalize(
            self.data_5(data_19),
            dim=-1
        )

        data_21 = torch.stack(
            data_15,
            dim=0
        ).mean(dim=0)

        return (
            data_18,
            data_9,
            data_11,
            data_20,
            data_21
        )

def data_21(data_22, data_23):
    data_24 = (
        data_22
        @
        data_23.T
    ) / value_9

    data_25 = torch.arange(
        data_24.shape[0],
        device=data_24.device
    )

    data_26 = (
        F.cross_entropy(
            data_24,
            data_25
        )
        +
        F.cross_entropy(
            data_24.T,
            data_25
        )
    ) / 2

    return data_26

def data_22(data_23):
    data_24 = data_23.clamp_min(
        1e-12
    )

    data_25 = -(
        data_24
        *
        torch.log2(data_24)
    ).sum(dim=-1)

    return data_25.mean(
        dim=(0, 2)
    )

def data_23(
    value_29,
    value_30,
    value_31
):
    if value_29 < value_30:
        return (
            value_29 + 1
        ) / value_30

    value_32 = (
        value_29 - value_30
    ) / max(
        1,
        value_31 - value_30 - 1
    )

    return 0.5 * (
        1
        +
        math.cos(
            math.pi * value_32
        )
    )

def data_24(
    model_3,
    value_33=True
):
    torch.manual_seed(
        value_7
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            value_7
        )

    data_25 = torch.Generator()
    data_25.manual_seed(
        value_7
    )

    data_26 = DataLoader(
        data_16,
        batch_size=value_4,
        shuffle=True,
        generator=data_25
    )

    optimizer_3 = torch.optim.Adam(
        model_3.parameters(),
        lr=value_10,
        weight_decay=value_11
    )

    value_34 = (
        value_12
        *
        len(data_26)
    )

    value_35 = len(
        data_26
    )

    scheduler_3 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_3,
        lambda value_36:
            data_23(
                value_36,
                value_35,
                value_34
            )
    )

    data_27 = []
    value_37 = 0

    for value_38 in range(
        value_12
    ):
        model_3.train()

        data_28 = []
        data_29 = []
        data_30 = []
        data_31 = []

        for data_32 in data_26:
            data_33 = data_32[0].to(
                device_1
            )

            data_34 = data_18(
                data_33,
                1.0
            )

            data_35 = data_18(
                data_33,
                1.0
            )

            data_36 = model_3(
                data_34,
                0.0
            )

            data_37 = model_3(
                data_35,
                0.0
            )

            loss_1 = data_21(
                data_36[3],
                data_37[3]
            )

            if value_33:
                data_38 = model_3(
                    data_33,
                    value_8
                )

                loss_2 = (
                    (
                        data_38[0]
                        -
                        data_38[1]
                    ) ** 2
                ).mean(dim=-1)

                loss_2 = loss_2[
                    data_38[2]
                ].mean()

                data_39 = data_38[4]
            else:
                loss_2 = torch.zeros(
                    (),
                    device=device_1
                )

                data_39 = data_36[4]

            loss_3 = (
                loss_1 + loss_2
            )

            optimizer_3.zero_grad()
            loss_3.backward()
            optimizer_3.step()
            scheduler_3.step()

            value_37 += 1

            data_28.append(
                float(
                    loss_2.detach().cpu()
                )
            )

            data_29.append(
                float(
                    loss_1.detach().cpu()
                )
            )

            data_30.append(
                float(
                    loss_3.detach().cpu()
                )
            )

            data_31.append(
                data_22(
                    data_39.detach()
                ).cpu().numpy()
            )

        data_31 = np.asarray(
            data_31
        )

        data_27.append([
            value_38 + 1,
            np.mean(data_28),
            np.mean(data_29),
            np.mean(data_30),
            np.mean(data_31),
            np.mean(
                data_31,
                axis=0
            )
        ])

    return (
        data_27,
        optimizer_3,
        value_37
    )

torch.manual_seed(
    value_7
)

model_1 = data_20().to(
    device_1
)

data_25, optimizer_1, value_39 = data_24(
    model_1,
    True
)

value_40 = sum(
    data_26.numel()
    for data_26 in model_1.parameters()
    if data_26.requires_grad
)

print(
    "Device:",
    device_1
)

print(
    "Observations:",
    len(data_9)
)

print(
    "Train / validation / test:",
    len(data_12),
    len(data_13),
    len(data_14)
)

print(
    "Trainable parameters:",
    value_40
)

model_1.eval()

data_26 = DataLoader(
    TensorDataset(data_15),
    batch_size=value_4,
    shuffle=False
)

data_27 = []

with torch.no_grad():
    for data_28 in data_26:
        data_29 = data_28[0].to(
            device_1
        )

        data_30 = model_1(
            data_29,
            0.0
        )

        data_27.append(
            data_30[3].cpu().numpy()
        )

data_27 = np.concatenate(
    data_27,
    axis=0
)

data_28 = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=value_7
).fit_transform(data_27)

figure_3 = plt.figure(
    figsize=(8, 6)
)

for value_41 in np.unique(
    data_10
):
    data_29 = (
        data_10 == value_41
    )

    plt.scatter(
        data_28[data_29, 0],
        data_28[data_29, 1],
        s=10,
        alpha=0.7,
        label=f"Group {value_41 + 1}"
    )

plt.xlabel(
    "Dimension 1"
)

plt.ylabel(
    "Dimension 2"
)

plt.title(
    "Figure 3"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()
plt.show()

data_30 = data_27[:64]

data_30 = data_30 / np.maximum(
    np.linalg.norm(
        data_30,
        axis=1,
        keepdims=True
    ),
    1e-12
)

data_31 = (
    data_30
    @
    data_30.T
)

figure_4 = plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    data_31,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.xlabel(
    "Observation"
)

plt.ylabel(
    "Observation"
)

plt.title(
    "Figure 4"
)

plt.colorbar()
plt.tight_layout()
plt.show()

data_32 = DataLoader(
    data_17,
    batch_size=value_4,
    shuffle=False
)

data_33 = next(
    iter(data_32)
)[0].to(device_1)

data_34 = np.linspace(
    0.0,
    1.0,
    6
)

data_35 = []

model_1.eval()

with torch.no_grad():
    for value_42 in data_34:
        torch.manual_seed(
            999
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                999
            )

        data_36 = data_18(
            data_33,
            float(value_42)
        )

        data_37 = data_18(
            data_33,
            float(value_42)
        )

        data_38 = model_1(
            data_36,
            0.0
        )[3]

        data_39 = model_1(
            data_37,
            0.0
        )[3]

        loss_4 = data_21(
            data_38,
            data_39
        )

        value_43 = (
            np.log(
                len(data_33)
            )
            -
            float(
                loss_4.cpu()
            )
        )

        data_35.append(
            value_43
        )

figure_5 = plt.figure(
    figsize=(7, 5)
)

plt.plot(
    data_34,
    data_35,
    marker="o"
)

plt.xlabel(
    "Augmentation strength"
)

plt.ylabel(
    "InfoNCE proxy"
)

plt.title(
    "Figure 5"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()

data_40 = data_25[0]
data_41 = data_25[-1]
data_42 = data_41[5]

data_43 = [
    [
        "Epoch 1",
        data_40[1],
        data_40[2],
        data_40[3],
        data_40[4]
    ],
    [
        "Epoch 4",
        data_41[1],
        data_41[2],
        data_41[3],
        data_41[4]
    ],
    [
        "Head average",
        data_42[0],
        data_42[1],
        data_42[2],
        data_42[3]
    ]
]

table_2 = pd.DataFrame(
    data_43,
    columns=[
        "Item",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)

display(
    table_2
)

def data_44(
    data_45,
    value_44
):
    data_46 = torch.complex(
        data_45[..., 0],
        data_45[..., 1]
    )

    data_47 = torch.mean(
        torch.abs(data_46) ** 2,
        dim=1
    )

    data_48 = data_47 / (
        10 ** (
            value_44 / 10
        )
    )

    data_49 = (
        torch.randn_like(
            data_46.real
        )
        +
        1j
        *
        torch.randn_like(
            data_46.real
        )
    )

    data_49 = data_49 * torch.sqrt(
        data_48[:, None] / 2
    )

    data_50 = (
        data_46 + data_49
    )

    return torch.stack(
        [
            data_50.real,
            data_50.imag
        ],
        dim=-1
    ).float()

def data_45(
    model_4,
    value_45
):
    torch.manual_seed(
        3333
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            3333
        )

    model_4.eval()

    data_46 = []

    with torch.no_grad():
        for data_47 in data_32:
            data_48 = data_47[0].to(
                device_1
            )

            data_49 = data_44(
                data_48,
                value_45
            )

            data_50 = model_4(
                data_49,
                value_8
            )

            data_51 = model_4.data_7(
                data_48
            )

            data_52 = F.cosine_similarity(
                data_50[0][data_50[2]],
                data_51[data_50[2]],
                dim=-1
            )

            data_46.extend(
                data_52.cpu().numpy()
            )

    return np.percentile(
        np.asarray(data_46),
        [10, 25, 50, 75]
    )

def data_46(
    model_5
):
    torch.manual_seed(
        2222
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            2222
        )

    model_5.eval()

    data_47 = []
    data_48 = []
    data_49 = []
    data_50 = []

    with torch.no_grad():
        for data_51 in data_32:
            data_52 = data_51[0].to(
                device_1
            )

            data_53 = model_5(
                data_52,
                value_8
            )

            data_54 = (
                (
                    data_53[0]
                    -
                    data_53[1]
                ) ** 2
            ).mean(dim=-1)

            data_54 = data_54[
                data_53[2]
            ].mean()

            data_55 = data_18(
                data_52,
                1.0
            )

            data_56 = data_18(
                data_52,
                1.0
            )

            data_57 = model_5(
                data_55,
                0.0
            )

            data_58 = model_5(
                data_56,
                0.0
            )

            loss_5 = data_21(
                data_57[3],
                data_58[3]
            )

            value_46 = torch.mean(
                torch.sum(
                    data_57[3]
                    *
                    data_58[3],
                    dim=1
                )
            )

            value_47 = torch.mean(
                data_22(
                    data_53[4]
                )
            )

            data_47.append(
                float(data_54.cpu())
            )

            data_48.append(
                float(loss_5.cpu())
            )

            data_49.append(
                float(
                    (1 - value_46).cpu()
                )
            )

            data_50.append(
                float(value_47.cpu())
            )

    return [
        np.mean(data_47),
        np.mean(data_48),
        np.mean(data_49),
        np.mean(data_50)
    ]

data_47 = data_45(
    model_1,
    -8
)

data_48 = data_45(
    model_1,
    2
)

torch.manual_seed(
    value_7
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        value_7
    )

model_2 = data_20().to(
    device_1
)

data_49, optimizer_2, value_48 = data_24(
    model_2,
    False
)

data_50 = data_46(
    model_1
)

data_51 = data_46(
    model_2
)

data_52 = [
    [
        "Condition 1",
        data_47[0],
        data_47[1],
        data_47[2],
        data_47[3]
    ],
    [
        "Condition 2",
        data_48[0],
        data_48[1],
        data_48[2],
        data_48[3]
    ],
    [
        "Configuration 1",
        data_50[0],
        data_50[1],
        data_50[2],
        data_50[3]
    ],
    [
        "Configuration 2",
        data_51[0],
        data_51[1],
        data_51[2],
        data_51[3]
    ]
]

table_3 = pd.DataFrame(
    data_52,
    columns=[
        "Item",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)

display(
    table_3
)

data_53 = {
    "model": model_1.state_dict(),
    "optimizer": optimizer_1.state_dict(),
    "epoch": value_12,
    "steps": value_39,
    "configuration": {
        "input_length": value_2,
        "patch_size": value_3,
        "embedding_dimension": value_4,
        "encoder_blocks": value_5,
        "attention_heads": value_6,
        "mask_ratio": value_8,
        "temperature": value_9,
        "learning_rate": value_10,
        "weight_decay": value_11
    }
}

torch.save(
    data_53,
    str(MODEL_DIR / "model_1.pt")
)

value_49 = os.path.getsize(
    str(MODEL_DIR / "model_1.pt")
) / (1024 ** 2)

data_54 = hashlib.sha1()

with open(
    str(MODEL_DIR / "model_1.pt"),
    "rb"
) as data_55:
    while True:
        data_56 = data_55.read(
            1024 * 1024
        )

        if not data_56:
            break

        data_54.update(
            data_56
        )

value_50 = data_54.hexdigest()

table_4 = pd.DataFrame(
    [
        ["Epoch", value_12],
        ["Steps", value_39],
        ["Size (MB)", value_49],
        ["SHA1", value_50]
    ],
    columns=[
        "Item",
        "Value"
    ]
)

