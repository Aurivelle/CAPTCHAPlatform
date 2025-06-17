PRESETS = {
    # 低難度：純淨圖像、字型佔比約 70%
    "Baseline": {
        "font_size": 42,  # 放大字型，適配 60x60
        "image_size": (60, 60),  # 統一所有 image_size
        "x_jitter": 1,
        "y_jitter": 1,
        "wave_amplitude": 0.0,
        "noise": {},
    },
    # 弱防禦：輕度抖動、微量 Gaussian noise + 隨機背景模糊
    "弱防禦": {
        "font_size": 44,
        "image_size": (60, 60),
        "x_jitter": 2,
        "y_jitter": 2,
        "wave_amplitude": 1.0,
        "noise": {
            "gaussian_noise": {"std": 10},
        },
    },
    # 中等防禦：中度抖動、旋轉、Cutout、彈性扭曲
    "中等防禦": {
        "font_size": 48,
        "image_size": (60, 60),
        "x_jitter": 3,
        "y_jitter": 3,
        "wave_amplitude": 2.0,
        "noise": {
            "gaussian_noise": {"std": 15},
            "rotate": {"angle_range": (-15, 15)},
            "cutout": {"num_patches": 2, "max_size": 0.25},
        },
    },
    # 強防禦：高強度扭曲、強旋轉、色彩亮度對比變化
    "強防禦": {
        "font_size": 52,
        "image_size": (60, 60),
        "x_jitter": 5,
        "y_jitter": 5,
        "wave_amplitude": 4.0,
        "noise": {
            "gaussian_noise": {"std": 20},
            "rotate": {"angle_range": (-30, 30)},
            "cutout": {"num_patches": 3, "max_size": 0.3},
            "brightness": {"factor_range": (0.7, 1.3)},
            "contrast": {"factor_range": (0.7, 1.3)},
        },
    },
    # 全方位極限防禦：整合所有perturber.py支援的干擾
    "全方位極限防禦": {
        "font_size": 56,
        "image_size": (60, 60),
        "x_jitter": 7,
        "y_jitter": 7,
        "wave_amplitude": 5.0,
        "noise": {
            "gaussian_noise": {"mean": 0, "std": 25},
            "laplace_noise": {"loc": 0, "scale": 15},
            "salt_pepper_noise": {"amount": 0.04, "s_vs_p": 0.5},
            "speckle_noise": {"std": 0.15},
            "rotate": {"angle_range": (-40, 40)},
            "affine_transform": {"max_shift": 0.12},
            "cutout": {"num_patches": 4, "max_size": 0.35},
            "occlusion_mask": {"num_shapes": 2, "max_size": 0.25},
            "brightness": {"factor_range": (0.5, 1.5)},
            "contrast": {"factor_range": (0.5, 1.5)},
            "jpeg_compression": {"quality_range": (20, 50)},
        },
    },
}
