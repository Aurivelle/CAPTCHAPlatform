# A Customizable CAPTCHA Generation and Evaluation Platform Against Machine Learning Attacks

## 📖 Project Overview
- 本專案實作並展示了一套完整的 CAPTCHA 攻防平台，包含：CAPTCHA 生成模組（多字元、抖動、變形、擾動）、攻擊模型：基於 Char-CNN 與 VGG 模型的字符識別、防禦介面--整合自訓練模型與 Tesseract OCR，比較並呈現識別準確度與字符錯誤率 (CER)、實時交互式平台：使用 Streamlit 提供即時演示與下載功能
- 此平台旨在強調：
  - 攻擊面：驗證自訓練 CNN/VGG 模型對抗不同雜訊與幾何變形的魯棒性
  - 防禦面：比較深度學習模型與傳統 OCR 在各種擾動條件下的效能差異
-  下載模型權重至 Model/ 目錄
  - char_cnn.pt
  - vgg16_char_best.pt

## 🚀 Usage
- 1. 啟動 Streamlit 介面
  - cd Src
  - python -m streamlit run app.py

## 📋 Script Descriptions
- Perturber.py
  - 支援多種雜訊：高斯、拉普拉斯、鹽與胡椒、斑點雜訊
  - 幾何變形：旋轉、仿射、遮罩
  - 顏色調整與 JPEG 壓縮模擬
- torch_char_cnn.py
  - 定義並訓練單字元 Char-CNN 網路
  - CPU/GPU 皆可啟動，預設訓練參數可達 90%+ train acc, 80%+ valid acc
- presets.py
  - 提供多組參數組合，快速生成測試集與批量推論設置
- evaluate.py
  - 加載訓練模型，對單字元圖片進行推論，計算 Accuracy 與 CER
  - 輸出與 Tesseract OCR 的效能比較
- metrics.py
  - 實作 Accuracy、Character Error Rate (CER) 指標
- data.py
  - 自動生成 CAPTCHA 圖片集，內含多字元支持
  - 可使用 presets 參數快速批量產生
- app.py
  - 透過 Streamlit 提供 Web 介面
  - 實時生成 CAPTCHA、模型推論、OCR 比較與結果下載
- VGG.ipynb
  - 使用 VGG16 架構訓練單字元分類
  - 實驗較強模型於本平台之應用潛力

