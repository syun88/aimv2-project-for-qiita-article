from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
import torch
import time
import cv2
import numpy as np

# YOLOモデルのロード
yolo_model = YOLO("yolov8n.pt")

# フォント設定
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
# query_text = ["pepsi can", "cola can", "sprite can", "fanta can"]
query_text = ["iphone", "ipad", "headphone", "Apple watch", "white book", "green book"]
# query_text = ["Bottle lying on its side","bottle"]
threshold = 0.3  # 類似度の閾値

# Webカメラの設定
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Webカメラを開けませんでした。")
    exit()

# FPS計算用
frame_count = 0
start_time = time.time()

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

# AIMv2の結果描画
def draw_aimv2_results(image, results):
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, label, score in results:
        draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
        draw.text((x1, y1 - 20), f"{label}: {score:.2f}", fill="blue", font=font)
    return image

print("Starting video stream...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        frame_count += 1

        # OpenCVのBGR画像をPillowのRGB画像に変換
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # YOLOによる物体検出
        results = yolo_model(frame)
        detections = results[0].boxes
        draw_yolo = draw_yolo_results(pil_image.copy(), detections)

        # AIMv2による条件一致領域の特定
        refined_results = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            region = pil_image.crop((x1, y1, x2, y2))

            # AIMv2で評価
            inputs = processor(images=region, text=query_text, return_tensors="pt", padding=True).to("cuda")
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)

            for i, score in enumerate(probs[0]):
                if score > threshold:
                    refined_results.append((x1, y1, x2, y2, query_text[i], score.item()))

        draw_aimv2 = draw_aimv2_results(pil_image.copy(), refined_results)

        # OpenCVで表示
        frame_with_yolo = cv2.cvtColor(np.array(draw_yolo), cv2.COLOR_RGB2BGR)
        frame_with_aimv2 = cv2.cvtColor(np.array(draw_aimv2), cv2.COLOR_RGB2BGR)

        cv2.imshow("YOLO Results", frame_with_yolo)
        cv2.imshow("AIMv2 Results", frame_with_aimv2)

        # FPS計算と表示
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame_with_aimv2, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AIMv2 Results", frame_with_aimv2)
        print(fps)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream ended.")
