import os
import random
import string
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from perturber import ImagePerturber
import math


def generate_random_text(length: int, charset: str) -> str:
    return "".join(random.choice(charset) for _ in range(length))


def get_font_paths(font_dir: str = "fonts") -> list[str]:
    paths = glob.glob(os.path.join(font_dir, "*.ttf"))
    if not paths:
        raise FileNotFoundError(f"No .ttf fonts found in {font_dir}")
    return paths


def generate_text_image(
    text: str,
    font_paths: list[str],
    font_size: int,
    image_size: tuple[int, int],
    bg_color: str,
    char_color: str,
    char_spacing: int,
    x_jitter: int = 0,
    y_jitter: int = 0,
    wave_amplitude: float = 0.0,
) -> Image.Image:
    img = Image.new("RGB", image_size, color=bg_color)
    draw = ImageDraw.Draw(img)
    font_path = random.choice(font_paths)
    font = ImageFont.truetype(font_path, font_size)

    def char_width(c: str) -> int:
        bbox = font.getbbox(c)
        return bbox[2] - bbox[0]

    widths = [char_width(c) for c in text]
    total_width = sum(widths) + char_spacing * (len(text) - 1)
    x_start = (image_size[0] - total_width) // 2
    y_start = (image_size[1] - font_size) // 2

    x = x_start
    for i, c in enumerate(text):
        dx = x + random.randint(-x_jitter, x_jitter)
        dy = (
            y_start
            + random.randint(-y_jitter, y_jitter)
            + int(wave_amplitude * math.sin(2 * math.pi * i / len(text)))
        )
        draw.text((dx, dy), c, font=font, fill=char_color)
        x += widths[i] + char_spacing
    return img


def generate_dataset(
    output_dir: str,
    n_samples: int,
    dataset_config: dict,
    noise_config: dict | None = None,
):
    output_path = Path(output_dir)
    clean_root = output_path / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    for ch in dataset_config["charset"]:
        (clean_root / ch).mkdir(exist_ok=True)
    label_path = output_path / "labels.txt"
    perturber = ImagePerturber(seed=dataset_config.get("seed", None))

    with open(label_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            text = generate_random_text(
                dataset_config["length"], dataset_config["charset"]
            )
            img = generate_text_image(
                text,
                dataset_config["font_paths"],
                dataset_config["font_size"],
                dataset_config["image_size"],
                dataset_config["bg_color"],
                dataset_config["char_color"],
                dataset_config["char_spacing"],
                x_jitter=dataset_config.get("x_jitter", 0),
                y_jitter=dataset_config.get("y_jitter", 0),
                wave_amplitude=dataset_config.get("wave_amplitude", 0.0),
            )
            if dataset_config.get("background_blur", False):
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

            label_dir = clean_root / text
            filename = f"{i:06d}.png"
            img.save(label_dir / filename)
            f.write(f"clean/{text}/{filename} {text}\n")

            if noise_config:
                for noise_name, params in noise_config.items():
                    noise_dir = output_path / noise_name / text
                    noise_dir.mkdir(parents=True, exist_ok=True)
                    noisy = perturber.apply(img, {noise_name: params})
                    noisy.save(noise_dir / filename)

    print(f"✅ Generated {n_samples} samples in '{output_dir}' (labels.txt saved)")


if __name__ == "__main__":
    dataset_cfg = {
        "length": 1,
        "charset": string.ascii_lowercase + string.digits,
        "font_paths": get_font_paths("fonts"),
        "font_size": 42,
        "image_size": (60, 60),
        "bg_color": "white",
        "char_color": "black",
        "char_spacing": 4,
        "seed": 42,
        "x_jitter": 5,
        "y_jitter": 5,
        "wave_amplitude": 2.0,
        "background_blur": True,
    }
    noise_cfg = {
        "gaussian_noise": {"mean": 0, "std": 10},
        "laplace_noise": {"loc": 0, "scale": 10},
        "salt_pepper_noise": {"amount": 0.02, "s_vs_p": 0.5},
        "speckle_noise": {"std": 0.1},
        "rotate": {"angle_range": (-10, 10)},
        "affine_transform": {"max_shift": 0.08},
        "cutout": {"num_patches": 1, "max_size": 0.2},
        "occlusion_mask": {"num_shapes": 1, "max_size": 0.2},
        "brightness": {"factor_range": (0.7, 1.3)},
        "contrast": {"factor_range": (0.7, 1.3)},
        "jpeg_compression": {"quality_range": (40, 80)},
    }
    generate_dataset(
        "dataset", n_samples=5000, dataset_config=dataset_cfg, noise_config=noise_cfg
    )
