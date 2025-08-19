#!/usr/bin/env bash
set -euo pipefail

# 元データセットのルート
SOURCE_ROOT="/mount/Oulu/cropped_mouth_mp4_phrase_pad"
# 出力先のベース
BASE_TARGET="/workspace/auto-avsr/oulu"

# 角度(ソース) と 出力ディレクトリ(ターゲット)の対応
# 1 -> video0, 3 -> video45, 5 -> video90
SRC_VIEW_IDS=(1 3 5)
TARGET_SUBDIRS=(video0 video45 video90)

# 出力先ディレクトリの作成
for sub in "${TARGET_SUBDIRS[@]}"; do
  mkdir -p "${BASE_TARGET}/${sub}"
done

# s* ディレクトリを順番に処理
find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r speaker_dir; do
  # 各角度ごとに処理
  for i in "${!SRC_VIEW_IDS[@]}"; do
    src_id="${SRC_VIEW_IDS[$i]}"
    target_dir="${BASE_TARGET}/${TARGET_SUBDIRS[$i]}"

    v_dir="${speaker_dir}/${src_id}"
    [[ -d "$v_dir" ]] || continue  # 角度フォルダがなければスキップ

    shopt -s nullglob
    for video in "$v_dir"/*.mp4; do
      filename="$(basename "$video")"
      # 絶対パスへ解決（Docker/Ubuntuなら readlink -f が使えます）
      src_abs="$(readlink -f "$video")"
      # 既存リンク/ファイルがあっても更新する (-n は既存シンボリックリンクを置換)
      ln -sfn "$src_abs" "$target_dir/$filename"
    done
    shopt -u nullglob
  done
done

echo "✅ Done: linked 1->video0, 3->video45, 5->video90"
