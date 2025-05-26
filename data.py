import os
import random
import string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from perturber import ImagePerturber
import math


def generate_random_text(length: int, charset: str) -> str:
    return "".join(random.choice(charset) for _ in range(length))


def generate_text_image(
    text: str,
    font_paths: list[str],
    font_size: int,
    image_size: tuple,
    bg_color: str = "white",
    char_color: str = "black",
    char_spacing: int = 4,
    x_jitter: int = 0,  # ← 新增每字水平抖動範圍（像素）
    y_jitter: int = 0,  # ← 新增每字垂直抖動範圍
    wave_amplitude: float = 0.0,  # ← 新增文字弧度或波形振幅
) -> Image.Image:
    img = Image.new("RGB", image_size, color=bg_color)
    # 隨機或由外部傳入決定要用哪個字型
    chosen_font = random.choice(font_paths)
    font = ImageFont.truetype(chosen_font, font_size)
    draw = ImageDraw.Draw(img)
    widths = []
    for c in text:
        bbox = draw.textbbox((0, 0), c, font=font)
        widths.append(bbox[2] - bbox[0])
    total_width = sum(widths) + char_spacing * (len(text) - 1)

    x = (image_size[0] - total_width) // 2
    y = (image_size[1] - font_size) // 2

    for i, c in enumerate(text):
        dx = x + random.randint(-x_jitter, x_jitter)
        dy = y + random.randint(-y_jitter, y_jitter)
        # 若要做波形扭曲，可對 dy 再加上 sin 函數偏移
        if wave_amplitude > 0:
            dy += int(wave_amplitude * math.sin(2 * math.pi * i / len(text)))
        draw.text((dx, dy), c, font=font, fill=char_color)
        x += widths[i] + char_spacing
    return img


def generate_dataset(
    output_dir: str,
    n_samples: int,
    dataset_config: dict,
    noise_config: dict | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_path = output_dir / "labels.txt"
    perturber = ImagePerturber(seed=dataset_config.get("seed", None))

    charset = dataset_config["charset"]
    length = dataset_config["length"]
    font_paths = dataset_config["font_paths"]
    font_size = dataset_config["font_size"]
    image_size = tuple(dataset_config["image_size"])
    bg_color = dataset_config.get("bg_color", "white")
    char_color = dataset_config.get("char_color", "black")
    char_spacing = dataset_config.get("char_spacing", 4)

    with label_path.open("w", encoding="utf-8") as f:
        for i in range(1, n_samples + 1):
            text = generate_random_text(length, charset)
            img = generate_text_image(
                text,
                font_paths,  # ← 傳入 list 而非單一路徑
                font_size,
                image_size,
                bg_color,
                char_color,
                char_spacing,
                x_jitter=dataset_config.get("x_jitter", 0),
                y_jitter=dataset_config.get("y_jitter", 0),
                wave_amplitude=dataset_config.get("wave_amplitude", 0.0),
            )
            filename = f"{i:06d}.png"
            img.save(output_dir / filename)
            f.write(f"{filename} {text}\n")

            if noise_config:
                for noise_name, params in noise_config.items():
                    subdir = output_dir / noise_name
                    subdir.mkdir(exist_ok=True)
                    noisy = perturber.apply(img, {noise_name: params})
                    noisy.save(subdir / filename)

    print(f"✅ Generated {n_samples} samples in '{output_dir}' (labels.txt saved)")


if __name__ == "__main__":
    dataset_cfg = {
        "length": 1,  # 單字元
        "charset": string.ascii_lowercase + string.digits,
        "font_paths": ["fonts/arial.ttf"],  # 或用 glob 掃描多字型
        "font_size": 42,
        "image_size": (60, 60),  # 單字元圖像可小一點
        "bg_color": "white",
        "char_color": "black",
        "char_spacing": 4,
        "seed": 42,
        "x_jitter": 5,
        "y_jitter": 5,
        "wave_amplitude": 2.0,
    }

    noise_cfg = {
        "gaussian_noise": {"std": 10},
        "cutout": {"num_patches": 1, "max_size": 0.2},
        "rotate": {"angle_range": (-10, 10)},
    }

    generate_dataset(
        output_dir="dataset",
        n_samples=500,
        dataset_config=dataset_cfg,
        noise_config=noise_cfg,
    )
