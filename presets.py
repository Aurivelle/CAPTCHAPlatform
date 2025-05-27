# -*- coding: utf-8 -*-
"""
三組示範防禦強度。image_size 與 font_size 保持 7~8 成比例，
並統一使用 'rotate' 這個 key（對應 perturber.ImagePerturber.rotate）
"""

PRESETS = {
    "Baseline": {  # 無任何擾動，用來驗證模型本身
        "font_size": 28,
        "image_size": (28, 28),
        "x_jitter": 0,
        "y_jitter": 0,
        "wave_amplitude": 0.0,
        "noise": {},  # 空 dict => 不加噪
    },
    "弱防禦": {
        "font_size": 30,
        "image_size": (32, 32),
        "x_jitter": 2,
        "y_jitter": 2,
        "wave_amplitude": 1.5,
        "noise": {
            "gaussian_noise": {"std": 10},
        },
    },
    "中等防禦": {
        "font_size": 34,
        "image_size": (36, 36),
        "x_jitter": 4,
        "y_jitter": 4,
        "wave_amplitude": 3.0,
        "noise": {
            "gaussian_noise": {"std": 15},
            "rotate": {"angle_range": (-15, 15)},
        },
    },
    "強防禦": {
        "font_size": 36,
        "image_size": (40, 40),
        "x_jitter": 6,
        "y_jitter": 6,
        "wave_amplitude": 5.0,
        "noise": {
            "gaussian_noise": {"std": 20},
            "rotate": {"angle_range": (-30, 30)},
            "cutout": {"num_patches": 3, "max_size": 0.3},
            "brightness": {"factor_range": (0.7, 1.3)},
            "contrast": {"factor_range": (0.7, 1.3)},
        },
    },
}
