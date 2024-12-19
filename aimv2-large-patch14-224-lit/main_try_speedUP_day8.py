from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
import torch
import time

# YOLOモデルのロード
yolo_model = YOLO("yolov8n.pt")

# 入力画像パス
image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/cola4.jpg"
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # フォントパス
font_size = 17
font = ImageFont.truetype(font_path, font_size)

# AIMv2モデルのロード
print("Loading AIMv2 model...")
start_load = time.time()
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True).to("cuda")
end_load = time.time()
print(f"Model loaded in {end_load - start_load:.2f} seconds")

# 条件テキスト
query_text = ["pepsi can", "cola can", "sprite can", "fanta can"]
threshold = 0.3  # 類似度の閾値

# 画像の読み込みとYOLOによる物体検出
image = Image.open(image_path).convert("RGB")
width, height = image.size
results = yolo_model(image_path)
detections = results[0].boxes

# YOLOの結果描画
def draw_yolo_results(image, detections):
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        label = yolo_model.names[int(detection.cls)]
        confidence = detection.conf.item()
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
        draw.text((x1, y1 - 20), f"{label}: {confidence:.2f}", fill="green", font=font)
    return image

draw_yolo = draw_yolo_results(image.copy(), detections)

# AIMv2による条件一致領域の特定
print("Processing AIMv2 detections...")
refined_results = []
aim_start = time.time()

for detection in detections:
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    region = image.crop((x1, y1, x2, y2))

    # AIMv2で評価
    inputs = processor(images=region, text=query_text, return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=-1)

    for i, score in enumerate(probs[0]):
        if score > threshold:
            refined_results.append((x1, y1, x2, y2, query_text[i], score.item()))

aim_end = time.time()

# AIMv2の結果描画
def draw_aimv2_results(image, results):
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, label, score in results:
        draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
        draw.text((x1, y1 - 20), f"{label}: {score:.2f}", fill="blue", font=font)
    return image

draw_aimv2 = draw_aimv2_results(image.copy(), refined_results)

# 処理結果の表示
print("Refined Results:")
for x1, y1, x2, y2, label, score in refined_results:
    print(f"Region ({x1}, {y1}, {x2}, {y2}) - {label}: {score:.2f}")

draw_yolo.show()  # YOLOの結果
draw_aimv2.show()  # AIMv2の結果

# 時間計測結果
print(f"AIMv2 Total Inference Time: {aim_end - aim_start:.4f} seconds")
