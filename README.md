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

Use  `tqdm ` in  `prepare_feature.py `  to visualize the progress of your code.
install tqdm
```commandline
pip install tqdm
```

# aimv2-project-for-qiita-article
Apple/aimv2 for-qiita-article
以下に、モデルのサイズが小さい順に並べたリストを示します。名前の中にある情報（`large` → `huge` → `1B` → `3B` など）や入力解像度（`224` → `336` → `448`）を基準にしています。

### サイズが小さい順のモデル
1. **`apple/aimv2-large-patch14-224`** <br>
    [Update at 2024/11/26 <br> aimv2-large-patch14-224の場所→](https://github.com/syun88/aimv2-project-for-qiita-article/tree/main/aimv2-large-patch14-224)
2. **`apple/aimv2-large-patch14-224-distilled`**
3. **`apple/aimv2-large-patch14-224-lit`**
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