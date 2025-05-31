import os
import random
import string
import tempfile
from pathlib import Path
from metrics import compute_accuracy, compute_CER

import streamlit as st
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from data import generate_text_image
from perturber import ImagePerturber

import torch
import torch.nn as nn
from torchvision import transforms
from torch_char_cnn import SimpleCNN

# ========== Streamlit App Configuration ==========
st.set_page_config(page_title="CAPTCHA Demo Page", layout="centered")
st.title("Customizable CAPTCHA Generation System")


# ========== Load Char-CNN Model ==========
@st.cache_resource
def load_charcnn(model_path: str = "char_cnn.pt", num_classes: int = 36):
    device = torch.device("cpu")
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


CHARSET = list(string.ascii_lowercase + string.digits)
char_cnn = load_charcnn(num_classes=len(CHARSET))
from torchvision import models


@st.cache_resource
def load_vgg16(model_path: str = "vgg16_char_best.pt", num_classes: int = 36):
    device = torch.device("cpu")
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


vgg16 = load_vgg16(num_classes=len(CHARSET))

# ========= Transforms & Predict Functions ==========
cnn_transform = transforms.Compose(
    [
        transforms.Resize((60, 60)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

vgg_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def vgg16_predict(img: Image.Image) -> str:
    x = vgg_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = vgg16(x)
    idx = out.argmax(1).item()
    return CHARSET[idx]


def charcnn_predict(img: Image.Image) -> str:
    x = cnn_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = char_cnn(x)
    idx = out.argmax(1).item()
    return CHARSET[idx]


def ocr_predict_with_conf(img: Image.Image):
    whitelist = "".join(CHARSET)
    data = pytesseract.image_to_data(
        img,
        config="--psm 10 -c tessedit_char_whitelist=" + whitelist,
        output_type=pytesseract.Output.DICT,
    )
    text = "".join(data["text"]).strip()
    confs = [float(c) for c in data["conf"] if c != "-1"]
    if confs:
        avg_conf = sum(confs) / len(confs)
    else:
        avg_conf = 0.0
    return text, avg_conf


def ocr_predict(img: Image.Image) -> str:
    t, _ = ocr_predict_with_conf(img)
    return t


def highlight_diff(pred: str, gt: str) -> str:
    out = []
    for p, g in zip(pred, gt):
        if p == g:
            out.append(p)
        else:
            out.append(f":red[{p}]")
    # 補尾差
    if len(pred) > len(gt):
        out += [f":red[{c}]" for c in pred[len(gt) :]]
    elif len(gt) > len(pred):
        out += [f":red[_]" for _ in gt[len(pred) :]]
    return "".join(out)


# ========== Sidebar: CAPTCHA Settings ==========
with st.sidebar:
    st.header("CAPTCHA Settings")
    text_input = st.text_input("Input text (leave blank → random)")
    if text_input:
        length = len(text_input)
        st.write(f"Text length: {length}")
    else:
        length = st.slider("Text length", 1, 5, 1)
    charset = string.ascii_lowercase + string.digits
    font_dir = Path("fonts")
    font_files = [str(p) for p in font_dir.glob("*.ttf")] or ["fonts/arial.ttf"]
    font_choice = st.selectbox("Font", ["Random"] + font_files)
    font_size = st.slider("Font size", 24, 60, 42)
    bg_color = st.color_picker("Background color", "#FFFFFF")
    char_color = st.color_picker("Text color", "#000000")
    st.markdown("---")
    st.subheader("Perturbation Settings")
    # Gaussian Noise
    apply_gauss = st.checkbox("Gaussian Noise")
    gauss_mean = st.slider("Gaussian mean", -30, 30, 0)
    gauss_std = st.slider("Gaussian σ", 5, 40, 15)
    # Laplace Noise
    apply_laplace = st.checkbox("Laplace Noise")
    laplace_loc = st.slider("Laplace loc", -30, 30, 0)
    laplace_scale = st.slider("Laplace scale", 5, 40, 20)
    # Salt & Pepper Noise
    apply_sap = st.checkbox("Salt & Pepper Noise")
    sap_amount = st.slider("S&P amount", 1, 10, 1) / 100
    sap_svp = st.slider("Salt vs Pepper", 0.0, 1.0, 0.5)
    # Speckle Noise
    apply_speckle = st.checkbox("Speckle Noise")
    speckle_std = st.slider("Speckle std", 0, 50, 10) / 100
    # Rotation
    apply_rot = st.checkbox("Rotation")
    rot_deg = st.slider("Rotation ±deg", 5, 45, 15)
    # Affine Transform
    apply_affine = st.checkbox("Affine Transform")
    affine_max_shift = st.slider("Max shift %", 0, 30, 10) / 100
    # Cutout
    apply_cut = st.checkbox("Cutout")
    cut_num = st.slider("Num patches", 1, 5, 2)
    cut_size = st.slider("Max patch %", 5, 40, 20) / 100
    # Occlusion Mask
    apply_occl = st.checkbox("Occlusion Mask")
    occl_num = st.slider("Num shapes", 1, 5, 1)
    occl_size = st.slider("Max size %", 5, 40, 20) / 100
    # Brightness
    apply_brightness = st.checkbox("Brightness")
    if apply_brightness:
        bright_min, bright_max = st.slider("Brightness range", 0.5, 2.0, (0.8, 1.2))
    # Contrast
    apply_contrast = st.checkbox("Contrast")
    if apply_contrast:
        cont_min, cont_max = st.slider("Contrast range", 0.5, 2.0, (0.8, 1.2))
    # JPEG Compression
    apply_jpeg = st.checkbox("JPEG Compression")
    jpeg_q = st.slider("Quality range", 10, 90, (30, 90))

# ========== Generate CAPTCHA Image ==========
if text_input:
    text = text_input
else:
    text = "".join(random.choice(charset) for _ in range(length))
font_paths = font_files if font_choice == "Random" else [font_choice]

max_chars = 5  # 160x60 大致能容納的字數上限
base_font_size = font_size
if length > max_chars:
    font_size = int(base_font_size * max_chars / length)

orig_img = generate_text_image(
    text=text,
    font_paths=font_paths,
    font_size=font_size,
    image_size=(160, 60),  # ← 與訓練保持一致
    bg_color=bg_color,
    char_color=char_color,
    char_spacing=4,
)

noise_cfg = {}
if apply_gauss:
    noise_cfg["gaussian_noise"] = {"mean": gauss_mean, "std": gauss_std}
if apply_laplace:
    noise_cfg["laplace_noise"] = {"loc": laplace_loc, "scale": laplace_scale}
if apply_sap:
    noise_cfg["salt_pepper_noise"] = {"amount": sap_amount, "s_vs_p": sap_svp}
if apply_speckle:
    noise_cfg["speckle_noise"] = {"std": speckle_std}
if apply_rot:
    noise_cfg["rotate"] = {"angle_range": (-rot_deg, rot_deg)}
if apply_affine:
    noise_cfg["affine_transform"] = {"max_shift": affine_max_shift}
if apply_cut:
    noise_cfg["cutout"] = {"num_patches": cut_num, "max_size": cut_size}
if apply_occl:
    noise_cfg["occlusion_mask"] = {"num_shapes": occl_num, "max_size": occl_size}
if apply_brightness:
    noise_cfg["brightness"] = {"factor_range": (bright_min, bright_max)}
if apply_contrast:
    noise_cfg["contrast"] = {"factor_range": (cont_min, cont_max)}
if apply_jpeg:
    noise_cfg["jpeg_compression"] = {"quality_range": jpeg_q}


perturber = ImagePerturber(seed=42)
noisy_img = perturber.apply(orig_img, noise_cfg) if noise_cfg else orig_img

# ========== Display CAPTCHA ==========
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original CAPTCHA")
    st.image(orig_img, use_container_width=True)
with col2:
    st.subheader("Perturbed CAPTCHA")
    st.image(noisy_img, use_container_width=True)
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    # 轉為灰階 numpy array
    orig_arr = np.array(orig_img.convert("L"))
    noisy_arr = np.array(noisy_img.convert("L"))
    # 計算 SSIM 與 PSNR
    ssim_val = ssim(orig_arr, noisy_arr)
    psnr_val = psnr(orig_arr, noisy_arr)
    st.info(
        f"SSIM 結構相似度: **{ssim_val:.3f}** | PSNR 峰值訊雜比: **{psnr_val:.2f} dB**"
    )
except ImportError:
    st.warning(
        "未安裝 scikit-image，無法計算 SSIM/PSNR。請執行 `pip install scikit-image`。"
    )
except Exception as e:
    st.warning(f"SSIM/PSNR 計算失敗: {e}")


st.markdown("---")
st.header("Model Inference Results")

if len(text) == 1:
    # 三個 baseline
    d1, d2, d3 = st.columns(3)
    with d1:
        try:
            pred_char = charcnn_predict(noisy_img)
            st.success(f"🖥️ Char-CNN predicts: **{pred_char}**")
        except Exception as e:
            st.error(f"Char-CNN inference failed: {e}")
    with d2:
        try:
            pred_vgg = vgg16_predict(noisy_img)
            st.info(f"🔬 VGG16 predicts: **{pred_vgg}**")
        except Exception as e:
            st.error(f"VGG16 inference failed: {e}")
    with d3:
        try:
            pred_ocr, ocr_conf = ocr_predict_with_conf(noisy_img)
            st.info(f"📝 OCR predicts: **{pred_ocr}** (confidence: {ocr_conf:.1f})")
        except Exception as e:
            st.error(f"OCR inference failed: {e}")

    # Ground truth & 比對
    if "pred_char" in locals() and "pred_ocr" in locals() and "pred_vgg" in locals():
        st.subheader("CAPTCHA 預測結果對比")
        st.write(f"Ground Truth: **{text}**")
        st.markdown(
            f"Char-CNN Output: {highlight_diff(pred_char, text)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"VGG16 Output: {highlight_diff(pred_vgg, text)}", unsafe_allow_html=True
        )
        st.markdown(
            f"Tesseract Output: {highlight_diff(pred_ocr, text)}",
            unsafe_allow_html=True,
        )
    from metrics import compute_similarity

    if "pred_char" in locals() and "pred_ocr" in locals() and "pred_vgg" in locals():
        st.subheader("CAPTCHA 預測相似度指標")
        cnn_sim = compute_similarity([pred_char], [text])
        vgg_sim = compute_similarity([pred_vgg], [text])
        ocr_sim = compute_similarity([pred_ocr], [text])
        st.write(f"Char-CNN  | Normalized Similarity: **{cnn_sim:.3f}**")
        st.write(f"VGG16     | Normalized Similarity: **{vgg_sim:.3f}**")
        st.write(f"Tesseract | Normalized Similarity: **{ocr_sim:.3f}**")
else:
    # 只顯示 Tesseract
    try:
        pred_ocr, ocr_conf = ocr_predict_with_conf(noisy_img)
        st.info(f"📝 OCR predicts: **{pred_ocr}** (confidence: {ocr_conf:.1f})")
    except Exception as e:
        st.error(f"OCR inference failed: {e}")

    st.subheader("Ground Truth")
    st.write(f"Ground Truth: **{text}**")
    st.markdown(
        f"Tesseract Output: {highlight_diff(pred_ocr, text)}", unsafe_allow_html=True
    )
    from metrics import compute_similarity

    ocr_sim = compute_similarity([pred_ocr], [text])
    st.write(f"Tesseract | Normalized Similarity: **{ocr_sim:.3f}**")

# ========== Download CAPTCHA ==========
st.markdown("---")
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / "captcha.png"
    noisy_img.save(tmp_path)
    st.download_button(
        label="💾 Download CAPTCHA",
        data=open(tmp_path, "rb").read(),
        file_name="captcha.png",
        mime="image/png",
    )

# ========== 批次上傳 & 評測 ==========
import tempfile
from metrics import evaluate_folder

st.markdown("---")
st.header("📂 批次上傳 & 評測")

uploaded_files = st.file_uploader(
    "上傳多張 CAPTCHA 圖片（PNG/JPG）",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# ====== 格式說明與範例 ======
st.markdown(
    """
**labels.txt 格式說明：**
- 每一行請輸入：`filename label`
- 例如：
    ```
    00001.png a
    00002.png g
    00003.png 8
    ```
- **檔名與標註以「空格」分隔。**
- 標註可為單字元（如 `a`），也可為多字元（如 `asdf`），但僅**全為單字元**時才同時評測 CNN/VGG baseline。
"""
)

# ====== 輸入框，含預設內容 ======
label_txt = st.text_area(
    "貼上 labels.txt 內容，每行 `filename label`",
    height=150,
    value="00001.png a\n00002.png b\n00003.png c",
)

# ====== 即時格式檢查 ======
if label_txt.strip():
    label_lines = [line for line in label_txt.strip().splitlines() if line.strip()]
    format_errors = []
    for idx, line in enumerate(label_lines):
        parts = line.strip().split()
        if len(parts) != 2:
            format_errors.append(idx + 1)
    if format_errors:
        st.warning(
            f"❗ 第 {format_errors} 行格式錯誤，應為：filename + 空格 + label（如 `00001.png a`）。"
        )

if st.button("開始批次評測"):
    if not uploaded_files:
        st.error("請先上傳檔案！")
    elif not label_txt.strip():
        st.error("請貼上 labels.txt 內容！")
    elif format_errors:
        st.error(f"labels.txt 有格式錯誤（第 {format_errors} 行），請修正後再提交！")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 儲存圖片
            for file in uploaded_files:
                open(os.path.join(tmpdir, file.name), "wb").write(file.getbuffer())
            # 2. 寫入 labels.txt
            lbl_path = os.path.join(tmpdir, "labels.txt")
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(label_txt.strip())
            # 3. 解析 labels 長度，判斷是否為單字元
            with open(lbl_path, "r", encoding="utf-8") as f:
                all_labels = [line.strip().split()[1] for line in f if line.strip()]
            single_char_mode = all(len(lbl) == 1 for lbl in all_labels)

            # 4. 呼叫 evaluate_folder
            if single_char_mode:
                res_cnn = evaluate_folder(charcnn_predict, tmpdir, lbl_path)
                res_vgg = evaluate_folder(vgg16_predict, tmpdir, lbl_path)
                res_ocr = evaluate_folder(ocr_predict, tmpdir, lbl_path)
                st.subheader("批次評測結果")
                st.write("🖥️ Char-CNN：", res_cnn)
                st.write("🔬 VGG16：", res_vgg)
                st.write("📝 Tesseract OCR：", res_ocr)
            else:
                res_ocr = evaluate_folder(ocr_predict, tmpdir, lbl_path)
                st.subheader("批次評測結果")
                st.write("📝 Tesseract OCR：", res_ocr)
