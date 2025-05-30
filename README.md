# CNS-Final-Project
### 程式們
**Note** : 如果有錯歡迎更正，期末頭腦有點不清楚。
#### Perturber.py
- 支援多種雜訊：高斯、拉普拉斯、鹽與胡椒、斑點雜訊
- 幾何變形：旋轉、仿射轉換、遮蔽形狀
- 顏色與壓縮：亮度與對比調整、JPEG 壓縮模擬
- 每種擾動方式都可以透過設定參數來控制效果
#### torch_char_cnn.py
- 可用於單字元 CAPTCHA 圖像分類任務的 CPU 訓練
- 用CPU訓練(當然也可以拿去用GPU訓練)
- 目前預設的參數，最後大概不會跑完，可以在train acc達到90幾%，valid acc達到80幾%。
- 可以比OCR表現好，以上兩點都是在基本的Dataset上去測試。
**Note** : 會利用原本train起來表現不錯，但實際放到```app.py```的前端的時候反而會看到預測爛掉去包裝這個平台的價值。
#### presets.py
- 算是一些參數，可以生產資料給下面的程式拿來測試train完的CNN Model的能力。
**Note** : 也是train起來表現不錯的論據之一。
#### evaluate.py
- 生產一些數據並拿來測試，算是Inference環節。
- 結果會是0.8, 0.9左右，同時會跟OCR做比較，而有比OCR高的成果。
#### metrics.py
- 目前主要是抓前兩個功能，也就是**acc**跟**CER**，另外兩個功能尚未安裝。
#### data.py
- 生成資料的模組
#### app.py
- CAPTCHA 圖像生成（支援多字元、抖動、變形、擾動）
- Char-CNN 模型推論(單字元)
- Tesseract OCR 模型推論
- 評估指標 (Accuracy、CER)
- 即時展示與下載功能
#### VGG.ipynb
- VGG是一種更強的訓練模型，但由於DataSet都是單字元所以有點尷尬。
- 字串的評分就交給OCR撐吧。
---
### 我的資料夾有但是Github上沒有的東西
**Note** : 原因是因為檔案太大了實在是push不上去QQ
- fonts/
    - 我電腦上抓的不同字型
    - 如果需要可以傳，但我想每個人的電腦上應該都有?主要是```.ttf```檔案。
- char_cnn.pt(也就是訓練好的模型)
  - 跑一次torch_char_cnn.py就可以用了
- vgg16_char_best.pt(也是訓練好的模型)
  - 跑一次VGG.ipynb就可以用了
- app.py 請用 ```python -m streamlit run app.py```
---
### 預想上的整體報告架構
- 基礎介紹
    - 動機
    - 問題
    - 攻擊模型
    - 一些名詞定義
    - Reference支撐
        - 應該不能只用之前報告的
        - 論文(雖然沒啥參考..)
- 實驗數據上的支持(強調攻擊面有一定實力)
- Streamlit攻防平台(強調防禦面能贏有一定實力的攻擊面)
- 待補充
    - 跟之前的Proposal對比看還少啥
    - 定義Threat Model
