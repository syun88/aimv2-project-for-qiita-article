import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとプロセッサのロード
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224", trust_remote_code=True).to(device)

# 特徴量抽出関数
def extract_features(image_dir, save_path):
    features_dict = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    print(f"Extracting features from {len(image_files)} images in {image_dir}...")
    for img_file in tqdm(image_files, desc="Extracting features", unit="image"):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # print(outputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().cpu()
            features_dict[img_file] = features
        except Exception as e:
            print(f"Error processing {img_file}: {e.__class__.__name__} - {e}")

    torch.save(features_dict, save_path)
    print(f"Features saved to {save_path}")

# メイン処理
if __name__ == "__main__":
    image_dir = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/coco_image/val2017"
    save_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/model/coco_features.pt"
    extract_features(image_dir, save_path)
