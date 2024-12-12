from ultralytics import YOLO
from PIL import Image, ImageDraw,ImageFont
from transformers import AutoProcessor, AutoModel
import torch
import time
# YOLOモデルのロード
yolo_model = YOLO("yolov8n.pt")
# image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/apple2.jpg"
# image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/cola2.jpg"
# image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/cola3.jpg"
image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/cola4.jpg"
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Ubuntu標準のフォントパス
font_size = 17  # テキストサイズ
font = ImageFont.truetype(font_path, font_size)# AIMv2モデルのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)

# 条件テキスト
# query_text = ["red apple","green apple"]
# query_text = ["cola bottle", "cola can", "cola glass"]
query_text = ["pepsi can", "cola can", "sprite can", "fanra can"]
# 類似度の結果を格納するリスト
high_score_regions = []
threshold = 0.8  # 類似度の閾値
# YOLOで物体検出
image = Image.open(image_path).convert("RGB")
width, height = image.size  # 元画像サイズを取得
results = yolo_model(image_path)
detections = results[0].boxes

# スケールを計算
# inference_shape = results[0].orig_shape  # YOLO推論時の画像サイズ
# scale_x = width / inference_shape[1]
# scale_y = height / inference_shape[0]

# YOLOの結果を描画
draw_yolo = image.copy()
draw = ImageDraw.Draw(draw_yolo)

for detection in detections:
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    # x1 = int(x1 * scale_x)
    # y1 = int(y1 * scale_y)
    # x2 = int(x2 * scale_x)
    # y2 = int(y2 * scale_y)

    label = yolo_model.names[int(detection.cls)]
    confidence = detection.conf.item()

    draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
    draw.text((x1, y1 - 20), f"{label}: {confidence:.2f}", fill="green", font=font)

# AIMv2で条件に一致する領域を特定
refined_results = []
aim_start = time.time()
n_list = []
for detection in detections:
    aim_start_n = time.time()
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    # x1 = int(x1 * scale_x)
    # y1 = int(y1 * scale_y)
    # x2 = int(x2 * scale_x)
    # y2 = int(y2 * scale_y)

    # YOLOで検出した領域を切り出し
    region = image.crop((x1, y1, x2, y2))
    
    # リサイズ
    # region_resized = region.resize((224, 224), Image.Resampling.LANCZOS)

    # デバッグ用にリサイズされた画像のサイズを確認
    # print(f"Region ({x1}, {y1}, {x2}, {y2}) Resized to: {region_resized.size}")

    # AIMv2で評価
    # inputs = processor(images=region_resized, text=query_text, return_tensors="pt", padding=True)
    inputs = processor(images=region, text=query_text, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)
    # 閾値を超える領域を記録
    for i, score in enumerate(probs[0]):
        if score > threshold:
            refined_results.append((x1, y1, x2, y2, query_text[i], score.item()))
        # print(f"Region ({x1}, {y1}, {x2}, {y2}) Score: {score:.2f}")
    aim_end_n = time.time()
    n_list.append(f" {aim_end_n - aim_start_n:.4f} seconds")

aim_end = time.time()
# AIMv2の結果を描画
draw_aimv2 = image.copy()
draw = ImageDraw.Draw(draw_aimv2)

print("Refined Results:")
for x1, y1, x2, y2, label, score in refined_results:
    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
    draw.text((x1, y1-20), f"{label}: {score:.2f}", fill="blue",font=font)
    print(f"Region ({x1}, {y1}, {x2}, {y2}) Score: {score:.2f}")
# 総時間計測: 終了
end_time = time.time()

for num,i in enumerate(n_list):
    print(f"object{num+1} cost {i}")
# 時間の出力
print(f"all AIMv2 Inference Time: {aim_end - aim_start:.4f} seconds")

# 結果を保存または表示
draw_yolo.show()  # YOLOの結果（緑色の枠）
draw_aimv2.show()  # AIMv2の結果（青色の枠）