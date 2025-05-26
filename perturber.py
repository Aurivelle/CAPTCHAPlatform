import numpy as np
from PIL import Image, ImageEnhance, ImageDraw


class ImagePerturber:
    """
    Noise, Distortions, and Color Augmentation for Images
    Example Usage :
        perturber = ImagePerturber(seed=42)
        cfg = {
            "gaussian_noise": {"std":15},
            "rotation": {"angle_range":(-15,15)},
            "cutout": {"num_patches":2, "max_size":0.2},
            "brightness": {"factor_range":(0.8,1.2)}
        }
        out = perturber.apply(img, cfg)
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    # ===== Low-level noise =====
    def gaussian_noise(
        self, img: Image.Image, mean: float = 0, std: float = 25
    ) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(mean, std, arr.shape)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def laplace_noise(
        self, img: Image.Image, loc: float = 0, scale: float = 20
    ) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.laplace(loc, scale, arr.shape)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def salt_pepper_noise(
        self, img: Image.Image, amount: float = 0.01, s_vs_p: float = 0.5
    ) -> Image.Image:
        arr = np.array(img)
        out = arr.copy()
        # salt
        num_salt = np.ceil(amount * arr.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
        out[tuple(coords)] = 255
        # pepper
        num_pepper = np.ceil(amount * arr.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
        out[tuple(coords)] = 0
        return Image.fromarray(out)

    def speckle_noise(self, img: Image.Image, std: float = 0.1) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * std
        out = np.clip(arr + arr * noise, 0, 1) * 255
        return Image.fromarray(out.astype(np.uint8))

    # ===== Geometric distortions =====
    def rotate(self, img: Image.Image, angle_range: tuple = (-15, 15)) -> Image.Image:
        angle = np.random.uniform(angle_range[0], angle_range[1])
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)

    def affine_transform(self, img: Image.Image, max_shift: float = 0.1) -> Image.Image:
        w, h = img.size
        # random shifting
        dx = np.random.uniform(-max_shift, max_shift) * w
        dy = np.random.uniform(-max_shift, max_shift) * h
        coeffs = (1, 0, dx, 0, 1, dy)
        return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)

    def cutout(
        self, img: Image.Image, num_patches: int = 1, max_size: float = 0.2
    ) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape[:2]
        out = arr.copy()
        for _ in range(num_patches):
            pw = int(np.random.uniform(0, max_size) * w)
            ph = int(np.random.uniform(0, max_size) * h)
            if (w - pw) <= 0 or (h - ph) <= 0:
                continue
            x = np.random.randint(0, w - pw)
            y = np.random.randint(0, h - ph)
            out[y : y + ph, x : x + pw] = 0
        return Image.fromarray(out)

    def occlusion_mask(
        self, img: Image.Image, num_shapes: int = 1, max_size: float = 0.2
    ) -> Image.Image:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = img.size
        for _ in range(num_shapes):
            size = np.random.uniform(0, max_size) * min(w, h)
            x0 = np.random.uniform(0, w - size)
            y0 = np.random.uniform(0, h - size)
            x1, y1 = x0 + size, y0 + size
            shape = [(x0, y0), (x1, y1)]
            draw.rectangle(shape, fill=(0))
        return out

    # ===== Color & style perturbation =====
    def brightness(
        self, img: Image.Image, factor_range: tuple = (0.8, 1.2)
    ) -> Image.Image:
        factor = np.random.uniform(factor_range[0], factor_range[1])
        return ImageEnhance.Brightness(img).enhance(factor)

    def contrast(
        self, img: Image.Image, factor_range: tuple = (0.8, 1.2)
    ) -> Image.Image:
        factor = np.random.uniform(factor_range[0], factor_range[1])
        return ImageEnhance.Contrast(img).enhance(factor)

    def jpeg_compression(
        self, img: Image.Image, quality_range: tuple = (30, 90)
    ) -> Image.Image:
        quality = int(np.random.uniform(quality_range[0], quality_range[1]))
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    # ===== Apply =====
    def apply(self, img: Image.Image, config: dict) -> Image.Image:
        """
        config 範例:
        {
            "gaussian_noise": {"std":15},
            "rotation": {"angle_range":(-15,15)},
            "cutout": {"num_patches":2, "max_size":0.2},
            "brightness": {"factor_range":(0.9,1.1)}
        }
        """
        if config is None or len(config) == 0:
            return img.copy()
        out = img.copy()
        for key, params in config.items():
            if hasattr(self, key):
                method = getattr(self, key)
                out = method(out, **params)
            else:
                raise ValueError(f"Unknown perturbation type: {key}")
        return out
