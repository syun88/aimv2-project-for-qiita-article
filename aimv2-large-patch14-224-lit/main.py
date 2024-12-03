import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModel
import TkEasyGUI as eg
from tqdm import tqdm

# モデルとプロセッサのロード
processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit", trust_remote_code=True)

# GUIを作成
layout = [
    [eg.Text("フォルダーを選択してください。")],
    [eg.Button("フォルダー選択", key="-SELECT_FOLDER-")],
    [eg.Text("検索キーワードを入力してください。")],
    [eg.Input("", key="-SEARCH_TEXT-")],
    [eg.Button("検索開始", key="-SEARCH-")],
    [eg.Button("閉じる", key="-CLOSE-")],
]

window = eg.Window("画像検索システム", layout=layout)

selected_dir = None

while True:
    event, values = window.read()

    if event == "-CLOSE-" or event is None:
        break

    if event == "-SELECT_FOLDER-":
        selected_dir = eg.popup_get_folder("フォルダーを選択してください。")
        if selected_dir:
            eg.popup(f"選択したフォルダー: {selected_dir}")

    if event == "-SEARCH-":
        if not selected_dir:
            eg.popup("先にフォルダーを選択してください。")
            continue

        search_text = values["-SEARCH_TEXT-"]
        if not search_text:
            eg.popup("検索キーワードを入力してください。")
            continue

        # 画像検索処理
        images = [f for f in os.listdir(selected_dir) if f.endswith(".jpg")]
        if not images:
            eg.popup("選択したフォルダーに画像がありません。")
            continue

        # スライディングウィンドウの設定
        window_size = 100
        stride = 50
        threshold = 0.85

        for img_file in images:
            image_path = os.path.join(selected_dir, img_file)
            image = Image.open(image_path)
            width, height = image.size

            # 類似度の結果を格納するリスト
            high_score_regions = []
            for y in tqdm(range(0, height - window_size + 1, stride)):
                for x in range(0, width - window_size + 1, stride):
            # for y in range(0, height - window_size + 1, stride):
            #     for x in range(0, width - window_size + 1, stride):
                    region = image.crop((x, y, x + window_size, y + window_size))
                    inputs = processor(
                        images=region,
                        text=[search_text],
                        return_tensors="pt",
                        padding=True,
                    )

                    # モデルで推論
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=-1)
                    score = probs[0, 0].item()

                    if score > threshold:
                        high_score_regions.append((x, y, window_size, window_size, search_text, score))

            # 結果を画像に描画
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)

            for region in high_score_regions:
                x, y, w, h, label, score = region
                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=1)
                draw.text((x, y - 10), f"{label}: {score:.2f}", fill="red")

            # 結果を表示（非ブロッキングモード）
            plt.figure(figsize=(10, 10))
            plt.imshow(draw_image)
            plt.axis("off")
            plt.title(f"結果: {img_file}")
            plt.show(block=False)

        # すべての画像を処理後にプロットを閉じる
        plt.close("all")

window.close()
