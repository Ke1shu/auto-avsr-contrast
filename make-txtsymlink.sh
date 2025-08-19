#!/bin/bash

# 元ファイルがあるディレクトリ
SRC_DIR="/mount/OuluVS2/OuluVS2/OuluVS2-zip/lsentence"

# シンボリックリンクを作るディレクトリ
DST_DIR="/workspace/auto-avsr/oulu/text"

# 出力先ディレクトリがなければ作成
mkdir -p "$DST_DIR"

# .txtファイルすべてを対象
for file in "$SRC_DIR"/*.txt; do
    filename=$(basename "$file")
    
    # 例: s1_u31.txt → s1_v1_u31.txt
    # アンダースコアの前に "v1_" を挿入
    #newname="${filename/_/_v1_}"
    newname=$filename

    # シンボリックリンク作成
    ln -s "$(realpath "$file")" "$DST_DIR/$newname"
    echo "[INFO] 作成: $DST_DIR/$newname"
done
