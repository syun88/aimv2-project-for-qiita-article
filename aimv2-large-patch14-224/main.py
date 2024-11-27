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

from tqdm import tqdm
import os
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとプロセッサのロード
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224", trust_remote_code=True).to(device)

# 検索対象画像のディレクトリ
image_dir = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/coco_image/val2017"
features_file = "coco_features.pt"

# 特徴量の保存またはロード
def extract_features(image_dir):
    features_dict = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for img_file in tqdm(image_files, desc="Extracting features", unit="image"):
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().cpu()
            features_dict[img_file] = features
        except Exception as e:
            print(f"Error processing {img_file}: {e.__class__.__name__} - {e}")
    return features_dict

if os.path.exists(features_file):
    print("Loading saved features...")
    features_dict = torch.load(features_file)
else:
    print("Extracting features from all images...")
    features_dict = extract_features(image_dir)
    torch.save(features_dict, features_file)
    print("Features saved to coco_features.pt")

# クエリ画像の特徴量
def get_query_features(query_image_path):
    query_image = Image.open(query_image_path).convert("RGB")
    query_inputs = processor(images=query_image, return_tensors="pt").to(device)
    query_outputs = model(**query_inputs)
    query_features = query_outputs.last_hidden_state.mean(dim=1).detach().cpu()
    return query_features


# クエリ画像を指定
query_image_path = "/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/cat.jpg" 
print("Extracting features from query image...")
query_features = get_query_features(query_image_path)

# 類似性計算
def find_most_similar_image(query_features, features_dict):
    best_match_file = None
    best_similarity = -1

    query_features = query_features.numpy()

    for img_file, img_features in features_dict.items():
        img_features = img_features.numpy()
        similarity = cosine_similarity(query_features, img_features)[0, 0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_file = img_file

    return best_match_file, best_similarity

print("Finding the most similar image...")
best_match_file, similarity_score = find_most_similar_image(query_features, features_dict)
print(f"Most similar image: {best_match_file}")
print(f"Similarity score: {similarity_score:.4f}")
