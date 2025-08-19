#!/bin/bash

# 元のデータセットルート
SOURCE_ROOT="/mount/OuluVS2/OuluVS2/OuluVS2-zip/cropped_mouth_mp4_phrase_pad"

# シンボリックリンクの出力先
TARGET_DIR="/workspace/auto-avsr/oulu/video90"

# 出力先ディレクトリを作成（なければ）
mkdir -p "$TARGET_DIR"

# s1〜s5 ディレクトリをループ
for speaker_dir in "$SOURCE_ROOT"/*; do
    v1_dir="$speaker_dir/5" #ここで角度指定
    if [ -d "$v1_dir" ]; then
        for video_file in "$v1_dir"/*.mp4; do
            [ -e "$video_file" ] || continue  # ファイルが存在しない場合スキップ
            filename=$(basename "$video_file")
            ln -s "$(realpath "$video_file")" "$TARGET_DIR/$filename"
        done
    fi
done
