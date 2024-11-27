import os
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel

# GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとプロセッサのロード
processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224", trust_remote_code=True).to(device)

# クエリ画像の特徴量を取得
def get_query_features(query_image_path):
    query_image = Image.open(query_image_path).convert("RGB")
    query_inputs = processor(images=query_image, return_tensors="pt").to(device)
    query_outputs = model(**query_inputs)
    # print(query_outputs)
    query_features = query_outputs.last_hidden_state.mean(dim=1).detach().cpu()
    return query_features

# 類似画像を検索
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

# メイン処理
if __name__ == "__main__":
    import argparse

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="Search for the most similar image")
    parser.add_argument("query_image", type=str, help="Path to the query image")
    parser.add_argument("--features", type=str, default="coco_features.pt", help="Path to the saved features file")
    args = parser.parse_args()

    # クエリ画像と特徴量のロード
    query_image_path = args.query_image
    features_path = args.features

    if not os.path.exists(features_path):
        print(f"Features file {features_path} not found. Run prepare_features.py first.")
        exit(1)

    print(f"Loading features from {features_path}...")
    features_dict = torch.load(features_path)

    print(f"Extracting features from query image: {query_image_path}...")
    query_features = get_query_features(query_image_path)

    print("Finding the most similar image...")
    best_match_file, similarity_score = find_most_similar_image(query_features, features_dict)
    print(f"Most similar image: {best_match_file}")
    print(f"Similarity score: {similarity_score:.4f}")
    # Image.open(f"/media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/coco_image/val2017{best_match_file}")
    # Image.open(query_image_path)