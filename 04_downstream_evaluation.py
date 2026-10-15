# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

data_56 = [
    "model_1",
    "data_15",
    "data_10",
    "data_12",
    "data_13",
    "data_14",
    "device_1"
]

data_57 = [
    data_58
    for data_58 in data_56
    if data_58 not in globals()
]

if len(data_57) > 0:
    raise RuntimeError(
        "Run Segment 2 first: " + ", ".join(data_57)
    )

data_58 = np.asarray(data_10)
data_59 = np.unique(data_58)

data_60 = {
    data_61: value_51
    for value_51, data_61 in enumerate(data_59)
}

data_61 = np.asarray([
    data_60[data_62]
    for data_62 in data_58
], dtype=np.int64)

value_51 = len(data_59)
value_52 = 128
value_53 = 1e-4
value_54 = 10
value_55 = value_54 * 4

data_62 = torch.tensor(
    data_61,
    dtype=torch.long
)

data_63 = [
    111,
    222,
    333
]

data_64 = [
    1.0,
    0.5,
    0.0
]

print("Classes:", value_51)
print("Train:", len(data_12))
print("Validation:", len(data_13))
print("Test:", len(data_14))
print("Train classes:", np.unique(data_61[data_12]))
print("Validation classes:", np.unique(data_61[data_13]))
print("Test classes:", np.unique(data_61[data_14]))

if len(np.unique(data_61[data_13])) != value_51:
    raise RuntimeError(
        "Validation partition does not contain every class."
    )

if len(np.unique(data_61[data_14])) != value_51:
    raise RuntimeError(
        "Test partition does not contain every class."
    )

class data_65(nn.Module):
    def __init__(self, data_66, value_56):
        super().__init__()

        self.data_1 = data_66

        self.data_2 = nn.Linear(
            128,
            value_56
        )

    def data_3(self, data_4):
        data_5 = self.data_1.data_7(
            data_4
        )

        data_5 = self.data_1.data_1(
            data_5
        )

        data_5 = (
            data_5
            +
            self.data_1.data_2
        )

        for data_6 in self.data_1.data_4:
            data_5, data_7 = data_6(
                data_5
            )

        data_8 = torch.mean(
            data_5,
            dim=1
        )

        return data_8

    def forward(self, data_4):
        data_5 = self.data_3(
            data_4
        )

        data_6 = self.data_2(
            data_5
        )

        return data_6, data_5

def data_66(data_67, value_57):
    for data_68 in data_67.parameters():
        data_68.requires_grad = True

    for data_68 in data_67.data_1.data_5.parameters():
        data_68.requires_grad = False

    for data_68 in data_67.data_1.data_6.parameters():
        data_68.requires_grad = False

    data_67.data_1.data_3.requires_grad = False

    value_58 = 0

    if value_57 >= 1.0:
        for data_68 in data_67.data_1.data_1.parameters():
            data_68.requires_grad = False

        data_67.data_1.data_2.requires_grad = False

        for data_68 in data_67.data_1.data_4.parameters():
            data_68.requires_grad = False

        value_58 = len(
            data_67.data_1.data_4
        )

    elif value_57 > 0:
        value_58 = int(
            np.ceil(
                len(data_67.data_1.data_4)
                *
                value_57
            )
        )

        for data_68 in list(
            data_67.data_1.data_4
        )[:value_58]:
            for data_69 in data_68.parameters():
                data_69.requires_grad = False

    return value_58

def data_67(data_68, value_59):
    if value_59 >= 1.0:
        data_68.data_1.eval()

    elif value_59 > 0:
        value_60 = int(
            np.ceil(
                len(data_68.data_1.data_4)
                *
                value_59
            )
        )

        for data_69 in list(
            data_68.data_1.data_4
        )[:value_60]:
            data_69.eval()

def data_68(data_69, data_70):
    data_69.eval()

    data_71 = []
    data_72 = []
    data_73 = []
    data_74 = 0.0

    with torch.no_grad():
        for data_75, data_76 in data_70:
            data_75 = data_75.to(
                device_1
            )

            data_76 = data_76.to(
                device_1
            )

            data_77, data_78 = data_69(
                data_75
            )

            data_79 = F.cross_entropy(
                data_77,
                data_76
            )

            data_74 += (
                float(data_79.cpu())
                *
                len(data_76)
            )

            data_71.append(
                data_77.cpu()
            )

            data_72.append(
                data_76.cpu()
            )

            data_73.append(
                data_78.cpu()
            )

    data_71 = torch.cat(
        data_71
    )

    data_72 = torch.cat(
        data_72
    )

    data_73 = torch.cat(
        data_73
    )

    data_80 = torch.argmax(
        data_71,
        dim=1
    )

    value_61 = accuracy_score(
        data_72.numpy(),
        data_80.numpy()
    )

    value_62 = (
        data_74
        /
        len(data_72)
    )

    return (
        value_61,
        value_62,
        data_71.numpy(),
        data_72.numpy(),
        data_73.numpy()
    )

def data_69(value_63, value_64):
    random.seed(
        value_63
    )

    np.random.seed(
        value_63
    )

    torch.manual_seed(
        value_63
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            value_63
        )

    data_70 = copy.deepcopy(
        model_1
    )

    data_70 = data_70.to(
        device_1
    )

    data_71 = data_65(
        data_70,
        value_51
    ).to(device_1)

    data_66(
        data_71,
        value_64
    )

    data_72 = TensorDataset(
        data_15[data_12],
        data_62[data_12]
    )

    data_73 = TensorDataset(
        data_15[data_13],
        data_62[data_13]
    )

    data_74 = torch.Generator()
    data_74.manual_seed(
        value_63
    )

    data_75 = DataLoader(
        data_72,
        batch_size=value_52,
        shuffle=True,
        generator=data_74
    )

    data_76 = DataLoader(
        data_73,
        batch_size=value_52,
        shuffle=False
    )

    data_77 = [
        data_78
        for data_78 in data_71.parameters()
        if data_78.requires_grad
    ]

    optimizer_3 = torch.optim.Adam(
        data_77,
        lr=value_53
    )

    value_65 = float("inf")
    value_66 = 0
    value_67 = 0

    data_78 = copy.deepcopy(
        data_71.state_dict()
    )

    data_79 = []

    for value_68 in range(
        value_55
    ):
        data_71.train()

        data_67(
            data_71,
            value_64
        )

        value_69 = 0.0
        value_70 = 0

        for data_80, data_81 in data_75:
            data_80 = data_80.to(
                device_1
            )

            data_81 = data_81.to(
                device_1
            )

            data_82, data_83 = data_71(
                data_80
            )

            loss_1 = F.cross_entropy(
                data_82,
                data_81
            )

            optimizer_3.zero_grad()
            loss_1.backward()
            optimizer_3.step()

            value_69 += (
                float(loss_1.detach().cpu())
                *
                len(data_81)
            )

            value_70 += len(
                data_81
            )

        data_84 = data_68(
            data_71,
            data_76
        )

        value_71 = (
            value_69
            /
            value_70
        )

        data_79.append([
            value_68 + 1,
            value_71,
            data_84[1],
            data_84[0]
        ])

        if data_84[1] < value_65:
            value_65 = data_84[1]
            value_66 = 0
            value_67 = value_68 + 1

            data_78 = copy.deepcopy(
                data_71.state_dict()
            )

        else:
            value_66 += 1

        if value_66 >= value_54:
            break

    data_71.load_state_dict(
        data_78
    )

    data_85 = data_68(
        data_71,
        data_76
    )

    return (
        data_71,
        data_85[0],
        data_85[1],
        data_79,
        value_67
    )

data_70 = []
data_71 = []
data_72 = []
data_73 = []

for value_72 in data_63:
    data_74 = data_69(
        value_72,
        0.0
    )

    data_70.append(
        data_74[1]
    )

    data_71.append(
        data_74[2]
    )

    data_72.append(
        data_74[4]
    )

    data_73.append({
        data_75: data_76.detach().cpu().clone()
        for data_75, data_76
        in data_74[0].state_dict().items()
    })

    print(
        "Seed",
        value_72,
        "validation accuracy:",
        f"{data_74[1]:.6f}",
        "best epoch:",
        data_74[4]
    )

    del data_74

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

value_73 = float(
    np.mean(data_70)
)

value_74 = float(
    np.std(
        data_70,
        ddof=1
    )
)

value_75 = int(
    np.argmin(
        np.abs(
            np.asarray(data_70)
            -
            value_73
        )
    )
)

value_76 = data_63[
    value_75
]

print(
    "Mean validation accuracy:",
    f"{value_73:.6f}"
)

print(
    "Validation standard deviation:",
    f"{value_74:.6f}"
)

print(
    "Representative seed:",
    value_76
)

data_75 = []
data_76 = {}
data_77 = {}

for value_77 in data_64:
    if value_77 == 0.0:
        data_78 = data_65(
            copy.deepcopy(model_1),
            value_51
        ).to(device_1)

        data_78.load_state_dict(
            data_73[value_75]
        )

        value_78 = data_70[
            value_75
        ]

        data_76[value_77] = data_78
        data_77[value_77] = value_78

    else:
        data_78 = data_69(
            value_76,
            value_77
        )

        data_76[value_77] = data_78[0]
        data_77[value_77] = data_78[1]

    print(
        "Setting",
        value_77,
        "validation accuracy:",
        f"{data_77[value_77]:.6f}"
    )

value_79 = data_77[
    1.0
]

data_79 = []

for value_80, value_81 in zip(
    data_63,
    data_70
):
    data_79.append([
        "Group 1",
        str(value_80),
        value_81,
        np.nan,
        np.nan
    ])

data_79.append([
    "Group 1",
    "Aggregate",
    value_73,
    np.nan,
    value_74
])

for value_80 in data_64:
    data_79.append([
        "Group 2",
        f"{value_80:.1f}",
        data_77[value_80],
        data_77[value_80] - value_79,
        np.nan
    ])

table_5 = pd.DataFrame(
    data_79,
    columns=[
        "Section",
        "Setting",
        "Accuracy",
        "Delta",
        "Dispersion"
    ]
)

display(
    table_5
)

model_3 = data_76[
    0.0
]

model_3.eval()

data_80 = DataLoader(
    TensorDataset(
        data_15[data_14],
        data_62[data_14]
    ),
    batch_size=value_52,
    shuffle=False
)

data_81 = data_68(
    model_3,
    data_80
)

data_82 = data_81[2]
data_83 = data_81[3]
data_84 = data_81[4]

data_85 = torch.softmax(
    torch.tensor(data_82),
    dim=1
).numpy()

data_86 = np.argmax(
    data_85,
    axis=1
)

data_87 = label_binarize(
    data_83,
    classes=np.arange(value_51)
)

if value_51 == 2:
    data_87 = np.column_stack([
        1 - data_87[:, 0],
        data_87[:, 0]
    ])

figure_6, data_88 = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

for value_82 in range(
    value_51
):
    data_89, data_90, data_91 = precision_recall_curve(
        data_87[:, value_82],
        data_85[:, value_82]
    )

    value_83 = average_precision_score(
        data_87[:, value_82],
        data_85[:, value_82]
    )

    data_88[0].plot(
        data_90,
        data_89,
        label=f"C{value_82} ({value_83:.3f})"
    )

    data_92, data_93, data_94 = roc_curve(
        data_87[:, value_82],
        data_85[:, value_82]
    )

    value_84 = roc_auc_score(
        data_87[:, value_82],
        data_85[:, value_82]
    )

    data_88[1].plot(
        data_92,
        data_93,
        label=f"C{value_82} ({value_84:.3f})"
    )

data_88[0].set_xlabel(
    "Recall"
)

data_88[0].set_ylabel(
    "Precision"
)

data_88[0].set_title(
    "Precision–Recall"
)

data_88[0].legend(
    fontsize=8
)

data_88[1].plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

data_88[1].set_xlabel(
    "False Positive Rate"
)

data_88[1].set_ylabel(
    "True Positive Rate"
)

data_88[1].set_title(
    "ROC"
)

data_88[1].legend(
    fontsize=8
)

figure_6.suptitle(
    "Figure 6"
)

figure_6.tight_layout()

plt.show()

data_95 = precision_recall_fscore_support(
    data_83,
    data_86,
    labels=np.arange(value_51),
    zero_division=0
)

data_96 = []

for value_85 in range(
    value_51
):
    if (
        data_95[0][value_85] == 0
        and
        data_95[1][value_85] == 0
        and
        data_95[2][value_85] == 0
    ):
        data_97 = "No recovered samples"
    else:
        data_97 = "Computed from predictions"

    data_96.append([
        f"Class {value_85}",
        data_95[0][value_85],
        data_95[1][value_85],
        data_95[2][value_85],
        data_97
    ])

data_98 = precision_recall_fscore_support(
    data_83,
    data_86,
    average="macro",
    zero_division=0
)

data_99 = precision_recall_fscore_support(
    data_83,
    data_86,
    average="weighted",
    zero_division=0
)

data_96.append([
    "Macro",
    data_98[0],
    data_98[1],
    data_98[2],
    "Mean across classes"
])

data_96.append([
    "Weighted",
    data_99[0],
    data_99[1],
    data_99[2],
    "Support weighted"
])

table_6 = pd.DataFrame(
    data_96,
    columns=[
        "Class",
        "Precision",
        "Recall",
        "F1",
        "Interpretation"
    ]
)

display(
    table_6
)

data_100 = data_65(
    copy.deepcopy(model_1),
    value_51
).to(device_1)

data_100.eval()

data_101 = DataLoader(
    TensorDataset(
        data_15,
        data_62
    ),
    batch_size=value_52,
    shuffle=False
)

data_102 = []
data_103 = []

with torch.no_grad():
    for data_104, data_105 in data_101:
        data_104 = data_104.to(
            device_1
        )

        data_106 = data_100.data_3(
            data_104
        )

        data_106 = F.normalize(
            data_106,
            dim=1
        )

        data_102.append(
            data_106.cpu().numpy()
        )

        data_103.append(
            data_105.numpy()
        )

data_102 = np.concatenate(
    data_102,
    axis=0
)

data_103 = np.concatenate(
    data_103,
    axis=0
)

data_107 = (
    data_102
    @
    data_102.T
)

data_108 = (
    data_103[:, None]
    ==
    data_103[None, :]
)

data_109 = ~np.eye(
    len(data_103),
    dtype=bool
)

data_110 = (
    data_108
    &
    data_109
)

data_111 = (
    (~data_108)
    &
    data_109
)

value_86 = float(
    np.mean(
        data_107[data_110]
    )
)

value_87 = float(
    np.mean(
        data_107[data_111]
    )
)

print(
    "Test accuracy:",
    f"{accuracy_score(data_83, data_86):.6f}"
)

print(
    "Intra-class cosine similarity:",
    f"{value_86:.6f}"
)

print(
    "Inter-class cosine similarity:",
    f"{value_87:.6f}"
)

print(
    "Similarity separation:",
    f"{value_86 - value_87:.6f}"
)

torch.save(
    model_3.state_dict(),
    str(MODEL_DIR / "model_3.pt")
)