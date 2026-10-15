# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import os
import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

data_112 = [
    "model_1",
    "data_19",
    "data_15",
    "data_62",
    "data_12",
    "data_13",
    "data_14",
    "device_1"
]

data_113 = [
    data_114
    for data_114 in data_112
    if data_114 not in globals()
]

if len(data_113) > 0:
    raise RuntimeError(
        "Run Segments 2 and 3 first: "
        +
        ", ".join(data_113)
    )

value_88 = 64
value_89 = 10
value_90 = 100
value_91 = 1e-4
value_92 = 10
value_93 = 40

data_114 = copy.deepcopy(
    model_1
).cpu()

data_114.eval()

torch.set_num_threads(1)

data_115 = data_15[
    data_14[:value_88]
].cpu()

if len(data_115) < value_88:
    value_94 = value_88 - len(
        data_115
    )

    data_115 = torch.cat(
        [
            data_115,
            data_15[
                data_14[:value_94]
            ].cpu()
        ],
        dim=0
    )

value_94 = sum(
    data_116.numel()
    for data_116 in data_114.parameters()
    if data_116.requires_grad
)

value_95 = sum(
    data_116.numel()
    *
    data_116.element_size()
    for data_116 in data_114.parameters()
)

value_96 = (
    value_95
    /
    (1024 ** 2)
)

data_116 = model_1.data_1
data_117 = model_1.data_4
data_118 = model_1.data_5
data_119 = model_1.data_6

value_97 = value_2 // value_3
value_98 = value_4
value_99 = value_3 * 2

value_100 = (
    2
    *
    value_97
    *
    data_116.in_features
    *
    data_116.out_features
)

value_101 = 0

for data_120 in data_117:
    value_102 = data_120.data_1.embed_dim
    value_103 = data_120.data_1.num_heads
    value_104 = value_102 // value_103

    value_105 = (
        6
        *
        value_97
        *
        value_102
        *
        value_102
    )

    value_106 = (
        2
        *
        value_97
        *
        value_102
        *
        value_102
    )

    value_107 = (
        2
        *
        value_103
        *
        value_97
        *
        value_97
        *
        value_104
    )

    value_108 = (
        2
        *
        value_103
        *
        value_97
        *
        value_97
        *
        value_104
    )

    value_109 = 0

    for data_121 in data_120.data_2:
        if isinstance(
            data_121,
            nn.Linear
        ):
            value_109 += (
                2
                *
                value_97
                *
                data_121.in_features
                *
                data_121.out_features
            )

    value_101 += (
        value_105
        +
        value_106
        +
        value_107
        +
        value_108
        +
        value_109
    )

value_110 = 0

for data_121 in data_118:
    if isinstance(
        data_121,
        nn.Linear
    ):
        value_110 += (
            2
            *
            data_121.in_features
            *
            data_121.out_features
        )

value_111 = (
    2
    *
    value_97
    *
    data_119.in_features
    *
    data_119.out_features
)

value_112 = int(
    value_100
    +
    value_101
    +
    value_110
    +
    value_111
)

data_122 = []

def data_123(
    data_124,
    data_125,
    data_126
):
    data_127 = data_126

    if isinstance(
        data_127,
        tuple
    ):
        data_127 = data_127[0]

    if torch.is_tensor(
        data_127
    ):
        data_122.append(
            data_127.numel()
            *
            data_127.element_size()
        )

data_128 = []

for data_129 in data_114.modules():
    if (
        isinstance(
            data_129,
            (
                nn.Linear,
                nn.LayerNorm,
                nn.MultiheadAttention
            )
        )
    ):
        data_128.append(
            data_129.register_forward_hook(
                data_123
            )
        )

with torch.no_grad():
    data_114(
        data_115[:1],
        0.0
    )

for data_129 in data_128:
    data_129.remove()

value_113 = (
    value_95
    +
    sum(data_122)
)

value_114 = (
    value_113
    /
    (1024 ** 2)
)

with torch.no_grad():
    for value_115 in range(
        value_89
    ):
        data_114(
            data_115,
            0.0
        )

data_130 = []

with torch.no_grad():
    for value_115 in range(
        value_90
    ):
        value_116 = time.perf_counter()

        data_114(
            data_115,
            0.0
        )

        value_117 = time.perf_counter()

        data_130.append(
            (
                value_117
                -
                value_116
            )
            *
            1000
        )

data_130 = np.asarray(
    data_130
)

value_118 = float(
    np.mean(data_130)
)

value_119 = float(
    np.std(
        data_130,
        ddof=1
    )
)

value_120 = float(
    np.percentile(
        data_130,
        95
    )
)

value_121 = (
    value_88
    /
    (
        value_118
        /
        1000
    )
)


class data_131(nn.Module):
    def __init__(
        self,
        data_132,
        value_122,
        value_123
    ):
        super().__init__()

        self.data_1 = copy.deepcopy(
            data_132.data_1
        )

        self.data_2 = nn.Parameter(
            data_132.data_2.detach().clone()
        )

        self.data_3 = nn.ModuleList()

        value_124 = len(
            data_132.data_4
        )

        for value_125 in range(
            value_122
        ):
            if value_125 < value_124:
                data_133 = copy.deepcopy(
                    data_132.data_4[
                        value_125
                    ]
                )

            else:
                data_133 = data_19()

            self.data_3.append(
                data_133
            )

        self.data_4 = nn.Linear(
            value_98,
            value_123
        )

    def forward(
        self,
        data_134
    ):
        data_135 = data_134.reshape(
            data_134.shape[0],
            value_97,
            value_3,
            2
        )

        data_135 = data_135.reshape(
            data_134.shape[0],
            value_97,
            value_99
        )

        data_135 = self.data_1(
            data_135
        )

        data_135 = (
            data_135
            +
            self.data_2
        )

        for data_136 in self.data_3:
            data_135, data_137 = data_136(
                data_135
            )

        data_138 = torch.mean(
            data_135,
            dim=1
        )

        data_139 = self.data_4(
            data_138
        )

        return (
            data_139,
            data_138
        )

def data_132(
    data_133,
    data_134
):
    data_133.eval()

    data_135 = []
    data_136 = []

    with torch.no_grad():
        for data_137, data_138 in data_134:
            data_137 = data_137.to(
                device_1
            )

            data_139, data_140 = data_133(
                data_137
            )

            data_135.append(
                data_139.cpu()
            )

            data_136.append(
                data_138
            )

    data_135 = torch.cat(
        data_135
    )

    data_136 = torch.cat(
        data_136
    )

    data_141 = torch.argmax(
        data_135,
        dim=1
    )

    return accuracy_score(
        data_136.numpy(),
        data_141.numpy()
    )

def data_133(
    value_126,
    value_127
):
    random.seed(
        value_127
    )

    np.random.seed(
        value_127
    )

    torch.manual_seed(
        value_127
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            value_127
        )

    data_134 = data_131(
        model_1,
        value_126,
        value_51
    ).to(device_1)

    data_135 = TensorDataset(
        data_15[data_12],
        data_62[data_12]
    )

    data_136 = TensorDataset(
        data_15[data_13],
        data_62[data_13]
    )

    data_137 = torch.Generator()
    data_137.manual_seed(
        value_127
    )

    data_138 = DataLoader(
        data_135,
        batch_size=128,
        shuffle=True,
        generator=data_137
    )

    data_139 = DataLoader(
        data_136,
        batch_size=128,
        shuffle=False
    )

    optimizer_4 = torch.optim.Adam(
        data_134.parameters(),
        lr=value_91
    )

    value_128 = float(
        "inf"
    )

    value_129 = 0

    data_140 = copy.deepcopy(
        data_134.state_dict()
    )

    for value_130 in range(
        value_93
    ):
        data_134.train()

        for data_141, data_142 in data_138:
            data_141 = data_141.to(
                device_1
            )

            data_142 = data_142.to(
                device_1
            )

            data_143, data_144 = data_134(
                data_141
            )

            loss_1 = F.cross_entropy(
                data_143,
                data_142
            )

            optimizer_4.zero_grad()
            loss_1.backward()
            optimizer_4.step()

        data_134.eval()

        value_131 = 0.0
        value_132 = 0

        with torch.no_grad():
            for data_141, data_142 in data_139:
                data_141 = data_141.to(
                    device_1
                )

                data_142 = data_142.to(
                    device_1
                )

                data_143, data_144 = data_134(
                    data_141
                )

                loss_2 = F.cross_entropy(
                    data_143,
                    data_142
                )

                value_131 += (
                    float(
                        loss_2.cpu()
                    )
                    *
                    len(data_142)
                )

                value_132 += len(
                    data_142
                )

        value_133 = (
            value_131
            /
            value_132
        )

        if value_133 < value_128:
            value_128 = value_133
            value_129 = 0

            data_140 = copy.deepcopy(
                data_134.state_dict()
            )

        else:
            value_129 += 1

        if value_129 >= value_92:
            break

    data_134.load_state_dict(
        data_140
    )

    value_134 = data_132(
        data_134,
        data_139
    )

    return (
        data_134,
        value_134
    )

def data_134(
    data_135
):
    data_136 = copy.deepcopy(
        data_135
    ).cpu()

    data_136.eval()

    data_137 = data_115

    with torch.no_grad():
        for value_135 in range(
            value_89
        ):
            data_136(
                data_137
            )

    data_138 = []

    with torch.no_grad():
        for value_135 in range(
            value_90
        ):
            value_136 = time.perf_counter()

            data_136(
                data_137
            )

            value_137 = time.perf_counter()

            data_138.append(
                (
                    value_137
                    -
                    value_136
                )
                *
                1000
            )

    return (
        float(
            np.mean(data_138)
        ),
        float(
            np.std(
                data_138,
                ddof=1
            )
        )
    )

data_135 = [
    2,
    3,
    4
]

data_136 = []

for value_138 in data_135:
    data_137, value_139 = data_133(
        value_138,
        111
    )

    data_138 = data_134(
        data_137
    )

    value_140 = sum(
        data_139.numel()
        for data_139 in data_137.parameters()
        if data_139.requires_grad
    )

    data_136.append([
        value_138,
        value_139,
        data_138[0],
        data_138[1],
        value_140
    ])

    print(
        "Depth:",
        value_138,
        "Accuracy:",
        f"{value_139:.6f}",
        "Latency:",
        f"{data_138[0]:.6f}",
        "Parameters:",
        value_140
    )

    del data_137

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

table_8 = pd.DataFrame(
    data_136,
    columns=[
        "Setting",
        "Accuracy",
        "Latency",
        "Dispersion",
        "Parameters"
    ]
)

display(
    table_8
)

figure_7 = plt.figure(
    figsize=(7, 5)
)

for data_140 in data_136:
    plt.scatter(
        data_140[2],
        data_140[1],
        s=70
    )

    plt.text(
        data_140[2],
        data_140[1],
        f" {data_140[0]}"
    )

plt.xlabel(
    "Latency (ms)"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Figure 7"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()
