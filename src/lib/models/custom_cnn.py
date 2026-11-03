"""Modelo A: CNN propia para embeddings faciales.

Entrada : tensor (N, 3, 112, 112), RGB, normalizado como ArcFace: (x - 127.5) / 127.5
          (el MISMO preprocesamiento que `lib.face_pipeline.preprocess_aligned_face`).
Salida  : embedding (N, embedding_dim) normalizado L2.

El modelo se exporta a ONNX para el backend (`MODEL_NAME=custom_cnn.onnx`), por lo que esta
clase solo hace falta para entrenar; el backend NO la importa.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Bloque residual: dos convoluciones 3x3 con BatchNorm y PReLU."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.act1 = nn.PReLU(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act2 = nn.PReLU(c_out)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class FaceEmbeddingNet(nn.Module):
    """ResNet liviana: stem + 4 etapas residuales + (pooling) + Dropout + Linear + BatchNorm1d.

    pool="gap"     : global average pooling (menos parámetros, más robusto con pocos datos).
    pool="flatten" : aplana el mapa 7x7 (conserva la posición espacial; más parámetros).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks: tuple[int, ...] = (1, 1, 1, 1),
        dropout: float = 0.3,
        pool: str = "gap",
        input_size: int = 112,
    ) -> None:
        super().__init__()
        assert pool in ("gap", "flatten")
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.PReLU(widths[0])
        )
        layers: list[nn.Module] = []
        c_in = widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(c_in, w, stride))
                c_in = w
        self.stages = nn.Sequential(*layers)

        if pool == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = widths[-1]
        else:
            self.pool = nn.Identity()
            side = input_size // 2 // (2 ** (len(widths) - 1))
            feat_dim = widths[-1] * side * side
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat_dim, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )
        self.embedding_dim = embedding_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.head(self.pool(x))
        return F.normalize(x, dim=1)


class ArcMarginProduct(nn.Module):
    """Cabeza de clasificación con margen angular aditivo (ArcFace).

    m = 0  -> softmax sobre cosenos escalados (clasificación + embedding, enfoque simple).
    m > 0  -> ArcFace: exige que el ángulo a la clase correcta sea al menos `m` radianes menor.
    Solo se usa para entrenar; se descarta al exportar.
    """

    def __init__(self, in_features: int, n_classes: int, s: float = 30.0, m: float = 0.3) -> None:
        super().__init__()
        self.s, self.m, self.n_classes = s, m, n_classes
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(emb), F.normalize(self.weight))

    def logits(self, emb: torch.Tensor) -> torch.Tensor:
        """Logits SIN margen (para medir accuracy y loss de validación)."""
        return self.s * self.cosine(emb)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(emb).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.m == 0:
            return self.s * cos
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self.cos_m - sin * self.sin_m          # cos(theta + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)  # evita que el margen "dé la vuelta"
        one_hot = F.one_hot(labels, self.n_classes).to(cos.dtype)
        return self.s * (one_hot * phi + (1.0 - one_hot) * cos)
