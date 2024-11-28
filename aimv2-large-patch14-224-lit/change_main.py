from PIL import Image
from transformers import AutoProcessor, AutoModel

# モデルとプロセッサのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)

# 入力データの準備
image = Image.open("/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/coco_image/val2017/000000010363.jpg")
text = ["cat", "dog", "bird"]  # 類似度を確認するテキストを複数用意
processor = AutoProcessor.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
)
model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
    trust_remote_code=True,
)

inputs = processor(
    images=image,
    text=text,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
print(outputs)
# 結果を表示
print("Predicted Probabilities:", probs)
for i, t in enumerate(text):
    print(f"{t}: {probs[0][i].item():.4f}")