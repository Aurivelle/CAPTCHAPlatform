import os
import random
import string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from perturber import ImagePerturber


def generate_random_text(length: int, charset: str) -> str:
    return "".join(random.choice(charset) for _ in range(length))


def generate_text_image(
    text: str,
    font_path: str,
    font_size: int,
    image_size: tuple,
    bg_color: str = "white",
    char_color: str = "black",
    char_spacing: int = 4,
) -> Image.Image:
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    widths = []
    for c in text:
        bbox = draw.textbbox((0, 0), c, font=font)
        widths.append(bbox[2] - bbox[0])
    total_width = sum(widths) + char_spacing * (len(text) - 1)

    x = (image_size[0] - total_width) // 2
    y = (image_size[1] - font_size) // 2

    for i, c in enumerate(text):
        draw.text((x, y), c, font=font, fill=char_color)
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
    font_path = dataset_config["font_paths"][0]
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
                font_path,
                font_size,
                image_size,
                bg_color,
                char_color,
                char_spacing,
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
        "length": 5,
        "charset": string.ascii_lowercase + string.digits,
        "font_paths": ["fonts/arial.ttf"],  # 確保此路徑存在
        "font_size": 42,
        "image_size": (160, 60),
        "bg_color": "white",
        "char_color": "black",
        "char_spacing": 4,
        "seed": 42,
    }

    noise_cfg = {
        "gaussian_noise": {"std": 15},
        # "rotation": {"angle_range": (-15, 15)},
        # "salt_pepper_noise": {"amount": 0.02, "s_vs_p": 0.5},
    }

    generate_dataset(
        output_dir="dataset",
        n_samples=500,
        dataset_config=dataset_cfg,
        noise_config=noise_cfg,
    )
