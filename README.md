# APPLE 公式さGITHUB readme Link 
[APPLE GITHUB README](https://github.com/apple/ml-aim/blob/main/README.md)

## 公式のInstallation
**CUDA のバージョンに応じて適切なバージョンｄにｓインストールしてください。ここでは説明しない。多くの記事はこの世の中に存在している。**

### pytorch install 公式サイトに参照
[installation instructions](https://pytorch.org/get-started/locally/).
パケージのインストール

```commandline
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v1'
pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'
```

**MLXに対応も可能だそうです**

### MLXとはMLX is an array framework for machine learning on Apple silicon, brought to you by Apple machine learning research.
### MLXは、アップルの機械学習研究によってもたらされた、アップルのシリコン上での機械学習のための配列フレームワークです。
### つまりこれでAppleの**MAC シリコン M chip** で動作も可能です。
[MLX](https://ml-explore.github.io/mlx/) backend support for research and experimentation on Apple silicon.
To enable MLX support, simply run:
```commandline
pip install mlx
```

transformers のインストール
```commandline
pip install transformers
```

cuda version check 
```commandline
nvidia-smi
```
or
```commandline
nvcc --version
```

install `tqdm` `tqdm` in  `prepare_feature.py`  to visualize the progress of your code. 
```commandline
pip install tqdm
```

dowonload val2017
```commacdline
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip
``` 
## Qiita DAY1 
[Qitta DAY1](https://qiita.com/syun88/items/50c1d60d1516d5816773)

cocodataset のimageをモデル化
```commandline
python3 aimv2-large-patch14-224/prepare_features.py 
```
aimv2-large-patch14-224/image_search.py の起動のやり方のサンプルコマンド
```commandline
python3 aimv2-large-patch14-224/image_search.py /media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/test_search_image/gtr.jpg --features /media/syun/ssd02/python_learning/apple/qiita_project_AIMv2/model/coco_features.pt
```
python3 aimv2-large-patch14-224/image_search.py 類似度入力に使うcocodataset内に存在していない画像 --features /path/model/coco_features.pt
`類似度入力に使うcocodataset内に存在していない画像` と `/path/model/coco_features.pt` はpathに置き換え

## Qitta DAY2 
[Qitta DAY2](https://qiita.com/syun88/items/11089454e046fe5e3f4d)

AppleのAIMv2でマルチモーダル機能を活用編1「画像領域特徴量の抽出とテキストで画像領域の可視化」の起動コマンド
```commandline
python3 aimv2-large-patch14-224-lit/image_search_from_text_and_show.py 
```

# aimv2-project-for-qiita-article
Apple/aimv2 for-qiita-article
以下に、モデルのサイズが小さい順に並べたリストを示します。名前の中にある情報（`large` → `huge` → `1B` → `3B` など）や入力解像度（`224` → `336` → `448`）を基準にしています。

### サイズが小さい順のモデル
1. **`apple/aimv2-large-patch14-224`** <br>
    [Update at 2024/11/26 aimv2-large-patch14-224の場所→](https://github.com/syun88/aimv2-project-for-qiita-article/tree/main/aimv2-large-patch14-224)
2. **`apple/aimv2-large-patch14-224-distilled`**
3. **`apple/aimv2-large-patch14-224-lit`**<br>
    [Update at 2024/11/29 aimv2-large-patch14-224-litの場所→](https://github.com/syun88/aimv2-project-for-qiita-article/tree/main/aimv2-large-patch14-224-lit)
4. **`apple/aimv2-large-patch14-native`**
5. **`apple/aimv2-large-patch14-336`**
6. **`apple/aimv2-large-patch14-336-distilled`**
7. **`apple/aimv2-large-patch14-448`**
8. **`apple/aimv2-huge-patch14-224`**
9. **`apple/aimv2-huge-patch14-336`**
10. **`apple/aimv2-huge-patch14-448`**
11. **`apple/aimv2-1B-patch14-224`**
12. **`apple/aimv2-1B-patch14-336`**
13. **`apple/aimv2-1B-patch14-448`**
14. **`apple/aimv2-3B-patch14-224`**
15. **`apple/aimv2-3B-patch14-336`**
16. **`apple/aimv2-3B-patch14-448`**

### 基準
- 解像度 (`224 < 336 < 448`) が小さいものを優先。
- モデルサイズ (`large < huge < 1B < 3B`) を優先。