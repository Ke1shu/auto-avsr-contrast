import os
import sentencepiece as spm
import cv2
import re
from preparation.transforms import TextTransform  # ✅ 追加

# --- パス設定 ---
text_dir = "oulu/text"
video_dir = "oulu/video45"
output_csv = "oulu/labels/45/all-data.csv"
sp_model_path = "spm/unigram/unigram25.model"
dict_path = "spm/unigram/unigram25_units.txt"  # ✅ 追加

# --- TextTransform によるトークナイザ読み込み ✅ 更新済み
text_transform = TextTransform(sp_model_path=sp_model_path, dict_path=dict_path)

# --- 出力ディレクトリ作成 ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# --- CSV生成 ---
with open(output_csv, 'w') as out:
    for txt_file in sorted(os.listdir(text_dir)):
        if not txt_file.endswith(".txt"):
            continue

        match = re.match(r'(s\d+)_u(\d+)\.txt', txt_file)
        if not match:
            print(f"[WARN] Unexpected filename format: {txt_file}")
            continue

        sid = match.group(1)
        uid = match.group(2)

        video_candidates = [f for f in os.listdir(video_dir)
                            if re.match(fr'{sid}_v\d+_u{uid}\.mp4', f)]

        if not video_candidates:
            print(f"[WARN] No video found for {txt_file}")
            continue

        video_file = sorted(video_candidates)[0]
        video_path = os.path.join(video_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ✅ テキスト読み込み & TextTransform によるトークン化（修正済み）
        txt_path = os.path.join(text_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        token_ids = text_transform.tokenize(text).tolist()

        # ✅ トークンID検証（TextTransform使用時は不要な可能性あり、必要なら残す）
        if any(t < 0 for t in token_ids):
            print(f"[NG] {video_file}: Invalid token ID(s) in {token_ids}")
            continue

        # CSV出力
        rel_video_path = f"video45/{video_file}"
        token_str = ' '.join(map(str, token_ids))
        out.write(f".,{rel_video_path},{frame_count},{token_str}\n")
