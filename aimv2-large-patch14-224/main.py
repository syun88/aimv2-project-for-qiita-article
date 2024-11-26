# original code 
# import requests
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModel

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# processor = AutoImageProcessor.from_pretrained(
#     "apple/aimv2-large-patch14-native",
# )
# model = AutoModel.from_pretrained(
#     "apple/aimv2-large-patch14-native",
#     trust_remote_code=True,
# )

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)
# print(outputs.last_hidden_state.shape)


import requests
from PIL import Image
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

# 画像URLの指定
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# モデルとプロセッサのロード
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-native")
model = AutoModelForImageClassification.from_pretrained(
    "apple/aimv2-large-patch14-native", trust_remote_code=True
)

# 入力データの作成
inputs = processor(images=image, return_tensors="pt")

# モデルで推論
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()

# ラベル情報を取得
config = AutoConfig.from_pretrained("apple/aimv2-large-patch14-native")
id2label = config.id2label
predicted_label = id2label[predicted_class]

# PIL画像をOpenCV形式に変換
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 分類結果を画像に描画
font = cv2.FONT_HERSHEY_SIMPLEX
text = f"Prediction: {predicted_label}"
font_scale = 1
font_color = (0, 255, 0)  # 緑
thickness = 2

# テキストの位置を指定
text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
text_x = 10
text_y = 30 + text_size[1]

cv2.putText(image_cv, text, (text_x, text_y), font, font_scale, font_color, thickness)

# OpenCVで画像を表示
cv2.imshow("Result", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
