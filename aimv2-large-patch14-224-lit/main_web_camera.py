import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw,ImageFont
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np

# YOLOモデルのロード
# yolo_model = YOLO("yolov8n.pt")  # 軽量モデル推奨
yolo_model = YOLO("yolov11n.pt")  # 軽量モデル推奨

# AIMv2モデルのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Ubuntu標準のフォントパス
font_size = 17  # テキストサイズ
font = ImageFont.truetype(font_path, font_size)# AIMv2モデルのロード
# 条件テキスト
# query_text = ["baseball", "green Pringles Chips can", " red Pringles Chips can","Tomato Soup can","Tomato"]
query_text = ["iphone", "ipad", "headphone","Apple watch","white book","green book"]

# 類似度の閾値
threshold = 0.8

# Webカメラを開く
cap = cv2.VideoCapture(0)  # デバイス番号を適宜設定
if not cap.isOpened():
    print("Webカメラを開けませんでした。")
    exit()

try:
    while True:
        # Webカメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        # OpenCVのBGR画像をPillowのRGB画像に変換
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = image.size

        # YOLOで物体検出
        results = yolo_model(frame)
        detections = results[0].boxes

        # スケールを計算
        inference_shape = results[0].orig_shape  # YOLO推論時の画像サイズ
        scale_x = width / inference_shape[1]
        scale_y = height / inference_shape[0]

        # YOLOの結果を描画
        draw_yolo = image.copy()
        draw = ImageDraw.Draw(draw_yolo)

        refined_regions = []
        regions = []

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label = yolo_model.names[int(detection.cls)]
            confidence = detection.conf.item()

            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
            draw.text((x1, y1 - 20), f"{label}: {confidence:.2f}", fill="green",font=font)

            # YOLOで検出した領域を切り出し（フィルタリングなし）
            regions.append(image.crop((x1, y1, x2, y2)))

        # バッチ処理でAIMv2に渡す
        if regions:
            inputs = processor(images=regions, text=query_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)

            for region_idx, region_probs in enumerate(probs):
                for i, score in enumerate(region_probs):
                    if score > threshold:
                        x1, y1, x2, y2 = map(int, detections[region_idx].xyxy[0])
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        refined_regions.append((x1, y1, x2, y2, query_text[i], score.item()))

        # AIMv2の結果を描画
        draw_aimv2 = image.copy()
        draw = ImageDraw.Draw(draw_aimv2)

        print("Refined Results:")
        for x1, y1, x2, y2, label, score in refined_regions:
            draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=3)
            draw.text((x1, y1 - 20), f"{label}: {score:.2f}", fill="blue",font=font)
            print(f"Region ({x1}, {y1}, {x2}, {y2}) Score: {score:.2f}")

        # 結果を表示
        yolo_frame = cv2.cvtColor(np.array(draw_yolo), cv2.COLOR_RGB2BGR)
        aimv2_frame = cv2.cvtColor(np.array(draw_aimv2), cv2.COLOR_RGB2BGR)

        # OpenCVでフレームを表示
        cv2.imshow("YOLO Results", yolo_frame)
        cv2.imshow("AIMv2 Results", aimv2_frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
