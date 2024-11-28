import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModel
import torch
from tqdm import tqdm

# モデルとプロセッサのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)

# 入力画像とテキスト
image = Image.open("/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/coco_image/val2017/000000010363.jpg")
text = ["cat", "dog", "bird"]

# スライディングウィンドウの設定
window_size = 100
stride = 50

# 類似度の結果を格納するリスト
high_score_regions = []
threshold = 0.85  # 類似度の閾値

# 画像サイズを取得
width, height = image.size

# 領域ごとの類似度計算
for y in tqdm(range(0, height - window_size + 1, stride)):
    for x in range(0, width - window_size + 1, stride):
        # 領域を切り出し
        region = image.crop((x, y, x + window_size, y + window_size))
        inputs = processor(images=region, text=text, return_tensors="pt", padding=True)
        
        # モデルで推論
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)

        # 閾値を超える領域を記録
        for i, score in enumerate(probs[0]):
            if score > threshold:
                high_score_regions.append((x, y, window_size, window_size, text[i], score.item()))

# 画像に枠を描画
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

for region in high_score_regions:
    x, y, w, h, label, score = region
    draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=1)  # 赤い枠を描画
    draw.text((x, y - 10), f"{label}: {score:.2f}", fill="red")  # ラベルとスコアを描画

# 結果を表示
plt.figure(figsize=(10, 10))
plt.imshow(draw_image)
plt.axis("off")
plt.show()
