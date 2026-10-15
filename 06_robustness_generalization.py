# Auto-extracted from wirlescomunication.ipynb. Run via ../../run_pipeline.py so shared state is preserved.

import os
import json
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
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data_141 = [
    "model_1",
    "data_15",
    "data_62",
    "data_12",
    "data_13",
    "data_14",
    "data_2",
    "device_1"
]

data_142 = [
    data_143
    for data_143 in data_141
    if data_143 not in globals()
]

if len(data_142) > 0:
    raise RuntimeError(
        "Run the earlier segments first: "
        +
        ", ".join(data_142)
    )

value_141 = len(
    data_15
)

value_142 = len(
    data_2
)

value_143 = len(
    np.unique(
        data_62.numpy()
    )
)

value_144 = 128
value_145 = 1e-4
value_146 = 10
value_147 = 40

data_143 = np.asarray(
    data_62.numpy(),
    dtype=np.int64
)

value_148 = value_141 // value_142
value_149 = value_141 % value_142

data_144 = []

for value_150 in range(
    value_142
):
    value_151 = (
        value_148
        +
        (
            1
            if value_150 < value_149
            else 0
        )
    )

    data_144.extend(
        [value_150] * value_151
    )

data_144 = np.asarray(
    data_144[:value_141],
    dtype=np.int64
)

data_145 = []

for data_146 in data_2:
    data_147 = (
        os.path.splitext(
            os.path.basename(data_146)
        )[0]
    )

    if "-D" in data_147:
        data_148 = data_147.rsplit(
            "-D",
            1
        )[0]
    else:
        data_148 = data_147

    data_145.append(
        data_148
    )

data_149 = np.asarray([
    data_145[data_150]
    for data_150 in data_144
])

data_151 = np.asarray([
    os.path.relpath(
        os.path.dirname(
            data_2[data_150]
        ),
        data_1
    )
    for data_150 in data_144
])

class data_152(nn.Module):
    def __init__(
        self,
        data_153,
        value_152
    ):
        super().__init__()

        self.data_1 = copy.deepcopy(
            data_153
        )

        self.data_2 = nn.Linear(
            128,
            value_152
        )

    def data_3(
        self,
        data_4
    ):
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

    def forward(
        self,
        data_4
    ):
        data_5 = self.data_3(
            data_4
        )

        data_6 = self.data_2(
            data_5
        )

        return (
            data_6,
            data_5
        )

def data_153(
    data_154,
    data_155
):
    data_154.eval()

    data_156 = []
    data_157 = []
    data_158 = []

    with torch.no_grad():
        for data_159, data_160 in data_155:
            data_159 = data_159.to(
                device_1
            )

            data_161, data_162 = data_154(
                data_159
            )

            data_156.append(
                data_161.cpu()
            )

            data_157.append(
                data_160.cpu()
            )

            data_158.append(
                data_162.cpu()
            )

    return (
        torch.cat(
            data_156
        ).numpy(),
        torch.cat(
            data_157
        ).numpy(),
        torch.cat(
            data_158
        ).numpy()
    )

def data_154(
    data_155,
    data_156,
    value_153
):
    random.seed(
        value_153
    )

    np.random.seed(
        value_153
    )

    torch.manual_seed(
        value_153
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            value_153
        )

    data_157 = data_152(
        model_1,
        value_143
    ).to(device_1)

    data_158 = TensorDataset(
        data_15[data_155],
        data_62[data_155]
    )

    data_159 = TensorDataset(
        data_15[data_156],
        data_62[data_156]
    )

    data_160 = torch.Generator()
    data_160.manual_seed(
        value_153
    )

    data_161 = DataLoader(
        data_158,
        batch_size=value_144,
        shuffle=True,
        generator=data_160
    )

    data_162 = DataLoader(
        data_159,
        batch_size=value_144,
        shuffle=False
    )

    optimizer_5 = torch.optim.Adam(
        data_157.parameters(),
        lr=value_145
    )

    value_154 = float(
        "inf"
    )

    value_155 = 0

    data_163 = copy.deepcopy(
        data_157.state_dict()
    )

    for value_156 in range(
        value_147
    ):
        data_157.train()

        for data_164, data_165 in data_161:
            data_164 = data_164.to(
                device_1
            )

            data_165 = data_165.to(
                device_1
            )

            data_166, data_167 = data_157(
                data_164
            )

            loss_1 = F.cross_entropy(
                data_166,
                data_165
            )

            optimizer_5.zero_grad()
            loss_1.backward()
            optimizer_5.step()

        data_157.eval()

        value_157 = 0.0
        value_158 = 0

        with torch.no_grad():
            for data_164, data_165 in data_162:
                data_164 = data_164.to(
                    device_1
                )

                data_165 = data_165.to(
                    device_1
                )

                data_166, data_167 = data_157(
                    data_164
                )

                loss_2 = F.cross_entropy(
                    data_166,
                    data_165
                )

                value_157 += (
                    float(
                        loss_2.cpu()
                    )
                    *
                    len(data_165)
                )

                value_158 += len(
                    data_165
                )

        value_159 = (
            value_157
            /
            value_158
        )

        if value_159 < value_154:
            value_154 = value_159
            value_155 = 0

            data_163 = copy.deepcopy(
                data_157.state_dict()
            )

        else:
            value_155 += 1

        if value_155 >= value_146:
            break

    data_157.load_state_dict(
        data_163
    )

    return data_157

data_168 = []

for value_160 in range(
    value_143
):
    data_169 = np.unique(
        data_144[
            data_143 == value_160
        ]
    )

    if len(data_169) > 1:
        data_168.append(
            data_169[-1]
        )

data_168 = np.asarray(
    data_168,
    dtype=np.int64
)

data_170 = np.isin(
    data_144,
    data_168
)

data_171 = np.asarray(
    data_12
)[
    ~data_170[
        np.asarray(data_12)
    ]
]

data_172 = np.asarray(
    data_13
)[
    ~data_170[
        np.asarray(data_13)
    ]
]

data_173 = np.asarray(
    data_14
)[
    ~data_170[
        np.asarray(data_14)
    ]
]

data_174 = np.where(
    data_170
)[0]

print(
    "Training samples:",
    len(data_171)
)

print(
    "Validation samples:",
    len(data_172)
)

print(
    "In-distribution test samples:",
    len(data_173)
)

print(
    "Held-out samples:",
    len(data_174)
)

model_4 = data_154(
    data_171,
    data_172,
    111
)

data_175 = DataLoader(
    TensorDataset(
        data_15[data_172],
        data_62[data_172]
    ),
    batch_size=value_144,
    shuffle=False
)

data_176 = DataLoader(
    TensorDataset(
        data_15[data_173],
        data_62[data_173]
    ),
    batch_size=value_144,
    shuffle=False
)

data_177 = DataLoader(
    TensorDataset(
        data_15[data_174],
        data_62[data_174]
    ),
    batch_size=value_144,
    shuffle=False
)

data_178 = data_153(
    model_4,
    data_175
)

data_179 = data_153(
    model_4,
    data_176
)

data_180 = data_153(
    model_4,
    data_177
)

def data_181(
    data_182,
    data_183
):
    data_184 = torch.tensor(
        data_182,
        dtype=torch.float32
    )

    data_185 = torch.tensor(
        data_183,
        dtype=torch.long
    )

    data_186 = torch.zeros(
        1,
        requires_grad=True
    )

    optimizer_6 = torch.optim.LBFGS(
        [data_186],
        max_iter=50
    )

    def data_187():
        optimizer_6.zero_grad()

        data_188 = torch.exp(
            data_186
        )

        loss_3 = F.cross_entropy(
            data_184 / data_188,
            data_185
        )

        loss_3.backward()

        return loss_3

    optimizer_6.step(
        data_187
    )

    return float(
        torch.exp(
            data_186.detach()
        )
    )

value_161 = data_181(
    data_178[0],
    data_178[1]
)

def data_182(
    data_183,
    data_184
):
    data_185 = torch.tensor(
        data_183,
        dtype=torch.float32
    )

    data_186 = torch.tensor(
        data_184,
        dtype=torch.long
    )

    data_187 = F.cross_entropy(
        data_185,
        data_186
    )

    return float(
        data_187
    )

def data_183(
    data_184,
    data_185
):
    data_186 = torch.softmax(
        torch.tensor(
            data_184,
            dtype=torch.float32
        ),
        dim=1
    ).numpy()

    data_187 = np.argmax(
        data_186,
        axis=1
    )

    data_188 = np.max(
        data_186,
        axis=1
    )

    data_189 = (
        data_187
        ==
        data_185
    ).astype(float)

    value_162 = max(
        2,
        int(
            np.ceil(
                np.sqrt(
                    len(data_185)
                )
            )
        )
    )

    data_190 = np.linspace(
        0,
        1,
        value_162 + 1
    )

    value_163 = 0.0

    for value_164 in range(
        value_162
    ):
        data_191 = (
            (data_188 > data_190[value_164])
            &
            (data_188 <= data_190[value_164 + 1])
        )

        if np.any(
            data_191
        ):
            value_163 += (
                np.mean(data_191)
                *
                abs(
                    np.mean(
                        data_189[data_191]
                    )
                    -
                    np.mean(
                        data_188[data_191]
                    )
                )
            )

    return value_163

data_192 = data_179[0]
data_193 = data_179[1]

data_194 = (
    data_192
    /
    value_161
)

value_165 = accuracy_score(
    data_193,
    np.argmax(
        data_192,
        axis=1
    )
)

value_166 = accuracy_score(
    data_193,
    np.argmax(
        data_194,
        axis=1
    )
)

value_167 = data_183(
    data_192,
    data_193
)

value_168 = data_183(
    data_194,
    data_193
)

value_169 = data_182(
    data_192,
    data_193
)

value_170 = data_182(
    data_194,
    data_193
)

data_195 = torch.softmax(
    torch.tensor(
        data_179[0]
    ),
    dim=1
).numpy()

data_196 = torch.softmax(
    torch.tensor(
        data_180[0]
    ),
    dim=1
).numpy()

data_197 = (
    1
    -
    np.max(
        data_195,
        axis=1
    )
)

data_198 = (
    1
    -
    np.max(
        data_196,
        axis=1
    )
)

data_199 = np.concatenate([
    np.zeros(
        len(data_197)
    ),
    np.ones(
        len(data_198)
    )
])

data_200 = np.concatenate([
    data_197,
    data_198
])

data_201, data_202, data_203 = roc_curve(
    data_199,
    data_200
)

value_171 = roc_auc_score(
    data_199,
    data_200
)

data_204, data_205, data_206 = precision_recall_curve(
    data_199,
    data_200
)

value_172 = average_precision_score(
    data_199,
    data_200
)

data_207 = np.where(
    data_202 >= 0.95
)[0]

value_173 = (
    float(
        data_201[
            data_207[0]
        ]
    )
    if len(data_207) > 0
    else np.nan
)

figure_8, data_208 = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

data_208[0].plot(
    data_201,
    data_202
)

data_208[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

data_208[0].set_xlabel(
    "False Positive Rate"
)

data_208[0].set_ylabel(
    "True Positive Rate"
)

data_208[0].set_title(
    f"ROC ({value_171:.3f})"
)

data_208[1].plot(
    data_205,
    data_204
)

data_208[1].set_xlabel(
    "Recall"
)

data_208[1].set_ylabel(
    "Precision"
)

data_208[1].set_title(
    f"PR ({value_172:.3f})"
)

figure_8.suptitle(
    "Figure 8"
)

figure_8.tight_layout()

plt.show()

table_9 = pd.DataFrame(
    [
        [
            "Setting 1",
            value_165,
            value_167,
            value_169,
            1.0,
            len(data_193)
        ],
        [
            "Setting 2",
            value_166,
            value_168,
            value_170,
            value_161,
            len(data_193)
        ],
        [
            "Setting 3",
            value_171,
            value_172,
            value_173,
            np.nan,
            len(data_199)
        ]
    ],
    columns=[
        "Setting",
        "C1",
        "C2",
        "C3",
        "C4",
        "N"
    ]
)

display(
    table_9
)

def data_209(
    data_210,
    value_174,
    value_175
):
    torch.manual_seed(
        value_175
    )

    data_211 = torch.complex(
        data_210[..., 0],
        data_210[..., 1]
    )

    data_212 = torch.mean(
        torch.abs(
            data_211
        ) ** 2,
        dim=1,
        keepdim=True
    )

    data_213 = (
        data_212
        /
        (
            10
            **
            (
                value_174
                /
                10
            )
        )
    )

    data_214 = (
        torch.randn_like(
            data_211.real
        )
        +
        1j
        *
        torch.randn_like(
            data_211.real
        )
    )

    data_214 = (
        data_214
        *
        torch.sqrt(
            data_213
            /
            2
        )
    )

    data_215 = (
        data_211
        +
        data_214
    )

    return torch.stack(
        [
            data_215.real,
            data_215.imag
        ],
        dim=-1
    ).float()

def data_210(
    model_5,
    data_211,
    data_212
):
    model_5.eval()

    data_213 = DataLoader(
        TensorDataset(
            data_211,
            data_212
        ),
        batch_size=value_144,
        shuffle=False
    )

    data_214 = []
    data_215 = []
    value_176 = 0.0

    with torch.no_grad():
        for data_216, data_217 in data_213:
            data_216 = data_216.to(
                device_1
            )

            data_217 = data_217.to(
                device_1
            )

            data_218, data_219 = model_5(
                data_216
            )

            data_220 = F.cross_entropy(
                data_218,
                data_217
            )

            value_176 += (
                float(
                    data_220.cpu()
                )
                *
                len(data_217)
            )

            data_214.append(
                torch.argmax(
                    data_218,
                    dim=1
                ).cpu()
            )

            data_215.append(
                data_217.cpu()
            )

    data_214 = torch.cat(
        data_214
    ).numpy()

    data_215 = torch.cat(
        data_215
    ).numpy()

    return (
        accuracy_score(
            data_215,
            data_214
        ),
        value_176
        /
        len(data_215)
    )

data_221 = np.linspace(
    10,
    30,
    3
)

data_222 = [
    111,
    222,
    333
]

data_223 = []

for value_177 in data_221:
    data_224 = []
    data_225 = []

    for value_178 in data_222:
        data_226 = data_209(
            data_15[data_173],
            float(value_177),
            value_178
        )

        data_227 = data_210(
            model_4,
            data_226,
            data_62[data_173]
        )

        data_224.append(
            data_227[0]
        )

        data_225.append(
            data_227[1]
        )

    data_223.append([
        f"{value_177:.1f}",
        np.mean(data_224),
        np.mean(data_225),
        np.std(
            data_224,
            ddof=1
        ),
        len(data_173)
    ])

data_228 = np.linspace(
    -0.2,
    0.2,
    3
)

for value_179 in data_228:
    value_180 = int(
        np.round(
            value_179
            *
            data_15.shape[1]
        )
    )

    data_229 = torch.roll(
        data_15[data_173],
        shifts=value_180,
        dims=1
    )

    data_230 = data_210(
        model_4,
        data_229,
        data_62[data_173]
    )

    data_223.append([
        f"{value_179:.2f}",
        data_230[0],
        data_230[1],
        0.0,
        len(data_173)
    ])

table_10 = pd.DataFrame(
    data_223,
    columns=[
        "Setting",
        "Accuracy",
        "NLL",
        "Dispersion",
        "N"
    ]
)

display(
    table_10
)

def data_231(
    data_232
):
    data_232.eval()

    data_233 = DataLoader(
        TensorDataset(
            data_15
        ),
        batch_size=value_144,
        shuffle=False
    )

    data_234 = []

    with torch.no_grad():
        for data_235 in data_233:
            data_236 = data_235[0].to(
                device_1
            )

            data_237 = data_232.data_7(
                data_236
            )

            data_237 = data_232.data_1(
                data_237
            )

            data_237 = (
                data_237
                +
                data_232.data_2
            )

            for data_238 in data_232.data_4:
                data_237, data_239 = data_238(
                    data_237
                )

            data_237 = torch.mean(
                data_237,
                dim=1
            )

            data_234.append(
                data_237.cpu().numpy()
            )

    return np.concatenate(
        data_234,
        axis=0
    )

data_240 = data_231(
    model_1
)

data_241 = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=111
).fit_transform(
    data_240
)

figure_9 = plt.figure(
    figsize=(9, 6)
)

for value_181 in np.unique(
    data_144
):
    data_242 = (
        data_144
        ==
        value_181
    )

    plt.scatter(
        data_241[
            data_242,
            0
        ],
        data_241[
            data_242,
            1
        ],
        s=9,
        alpha=0.65,
        label=f"G{value_181 + 1}"
    )

plt.xlabel(
    "Dimension 1"
)

plt.ylabel(
    "Dimension 2"
)

plt.title(
    "Figure 9"
)

plt.legend(
    fontsize=7,
    ncol=2
)

plt.tight_layout()

plt.show()

data_243 = []

for value_182 in np.unique(
    data_144
):
    data_244 = (
        data_144
        !=
        value_182
    )

    data_245 = (
        data_144
        ==
        value_182
    )

    data_246 = np.unique(
        data_143[data_244]
    )

    data_247 = np.unique(
        data_143[data_245]
    )

    if not set(
        data_247.tolist()
    ).issubset(
        set(
            data_246.tolist()
        )
    ):
        continue

    model_6 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000
        )
    )

    model_6.fit(
        data_240[data_244],
        data_143[data_244]
    )

    data_248 = model_6.predict(
        data_240[data_245]
    )

    value_183 = accuracy_score(
        data_143[data_245],
        data_248
    )

    value_184 = f1_score(
        data_143[data_245],
        data_248,
        average="macro",
        zero_division=0
    )

    data_243.append([
        "Group 1",
        str(value_182 + 1),
        value_183,
        value_184,
        int(
            np.sum(
                data_245
            )
        )
    ])

for data_249 in np.unique(
    data_149
):
    data_250 = (
        data_149
        !=
        data_249
    )

    data_251 = (
        data_149
        ==
        data_249
    )

    data_252 = np.unique(
        data_143[data_250]
    )

    data_253 = np.unique(
        data_143[data_251]
    )

    if not set(
        data_253.tolist()
    ).issubset(
        set(
            data_252.tolist()
        )
    ):
        continue

    model_7 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000
        )
    )

    model_7.fit(
        data_240[data_250],
        data_143[data_250]
    )

    data_254 = model_7.predict(
        data_240[data_251]
    )

    data_243.append([
        "Group 2",
        str(data_249),
        accuracy_score(
            data_143[data_251],
            data_254
        ),
        f1_score(
            data_143[data_251],
            data_254,
            average="macro",
            zero_division=0
        ),
        int(
            np.sum(
                data_251
            )
        )
    ])

table_11 = pd.DataFrame(
    data_243,
    columns=[
        "Section",
        "Setting",
        "Accuracy",
        "F1",
        "N"
    ]
)

display(
    table_11
)

data_255 = {
    "throughput": [],
    "latency": [],
    "drop": [],
    "action": []
}

def data_256(
    data_257
):
    if isinstance(
        data_257,
        dict
    ):
        for data_258, data_259 in data_257.items():
            data_260 = str(
                data_258
            ).lower()

            if isinstance(
                data_259,
                (int, float)
            ):
                if "throughput" in data_260:
                    data_255[
                        "throughput"
                    ].append(
                        float(data_259)
                    )

                if "latency" in data_260:
                    data_255[
                        "latency"
                    ].append(
                        float(data_259)
                    )

                if (
                    "drop" in data_260
                    or
                    "loss_rate" in data_260
                ):
                    data_255[
                        "drop"
                    ].append(
                        float(data_259)
                    )

                if "action" in data_260:
                    data_255[
                        "action"
                    ].append(
                        float(data_259)
                    )

            data_256(
                data_259
            )

    elif isinstance(
        data_257,
        list
    ):
        for data_258 in data_257:
            data_256(
                data_258
            )

for data_261 in data_2:
    data_262 = os.path.splitext(
        data_261
    )[0] + ".json"

    if os.path.exists(
        data_262
    ):
        try:
            with open(
                data_262,
                "r"
            ) as data_263:
                data_264 = json.load(
                    data_263
                )

            data_256(
                data_264
            )

        except:
            pass

print(
    "Control telemetry:",
    {
        data_265: len(
            data_266
        )
        for data_265, data_266
        in data_255.items()
    }
)

torch.save(
    model_4.state_dict(),
    str(MODEL_DIR / "model_4.pt")
)