import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModel
import torch

# YOLOモデルのロード
yolo_model = YOLO("yolov8n.pt")

# AIMv2モデルのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)

# 条件テキスト
query_text = ["pepsi can", "cola can", "sprite can", "fanta can"]

# 類似度の閾値
threshold = 0.8

# Webカメラを開く
cap = cv2.VideoCapture(0)

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

        refined_results = []

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label = yolo_model.names[int(detection.cls)]
            confidence = detection.conf.item()

            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
            draw.text((x1, y1 - 10), f"{label}: {confidence:.2f}", fill="green")

            # YOLOで検出した領域を切り出し
            region = image.crop((x1, y1, x2, y2))

            # リサイズ
            region_resized = region.resize((224, 224), Image.Resampling.LANCZOS)

            # AIMv2で評価
            inputs = processor(images=region_resized, text=query_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1)

            # 閾値を超える領域を記録
            for i, score in enumerate(probs[0]):
                if score > threshold:
                    refined_results.append((x1, y1, x2, y2, query_text[i], score.item()))

        # AIMv2の結果を描画
        draw_aimv2 = image.copy()
        draw = ImageDraw.Draw(draw_aimv2)

        print("Refined Results:")
        for x1, y1, x2, y2, label, score in refined_results:
            draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
            draw.text((x1, y1 - 10), f"{label}: {score:.2f}", fill="blue")
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
