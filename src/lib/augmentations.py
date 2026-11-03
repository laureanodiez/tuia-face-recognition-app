"""Augmentation para rostros alineados 112x112 (RGB uint8). Se aplica SOLO en train.

Es el mismo pipeline que se justificó en la sección 15 de 1-pre-processing.ipynb.
"""
from __future__ import annotations

import albumentations as A


def build_train_augmentation() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-10, 10), translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.3),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), p=0.3),
    ])
