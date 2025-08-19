import os
import re
import cv2
from preparation.transforms import TextTransform  # ✅ 既存の TextTransform を使用

# --- パス設定（必要に応じて書き換えてください） ---
text_dir    = "oulu/text"
video1_dir  = "oulu/video"   # ✅ v1 用ディレクトリ
video2_dir  = "oulu/video90"   # ✅ v2 用ディレクトリ
tg_dir      = "oulu/textgrid"     # ✅ TextGrid ディレクトリ
output_csv  = "oulu/labels/contrast/all-data.csv"

# --- SentencePiece / 辞書パス（既存どおり） ---
sp_model_path = "spm/unigram/unigram25.model"
dict_path     = "spm/unigram/unigram25_units.txt"

# --- TextTransform によるトークナイザ ---
text_transform = TextTransform(sp_model_path=sp_model_path, dict_path=dict_path)

# --- 出力ディレクトリ作成 ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

def _frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

def _rel_path(base_dir: str, fname: str) -> str:
    """出力用に 'ディレクトリ名/ファイル名' を作る（元コードの形式に合わせる）"""
    return f"{os.path.basename(base_dir)}/{fname}"

with open(output_csv, 'w', encoding='utf-8') as out:
    # ヘッダ行（必要に応じてアンコメント）
    # out.write(".,rel_path_v1,rel_path_v2,rel_path_tg,input_len_v1,input_len_v2,token_id\n")

    for txt_file in sorted(os.listdir(text_dir)):
        if not txt_file.endswith(".txt"):
            continue

        m = re.match(r'(s\d+)_u(\d+)\.txt', txt_file)
        if not m:
            print(f"[WARN] Unexpected filename format: {txt_file}")
            continue

        sid, uid = m.group(1), m.group(2)

        # --- video1 / video2 ファイル名を生成 ---
        vfile1 = f"{sid}_v1_u{uid}.mp4"
        vfile2 = f"{sid}_v5_u{uid}.mp4"

        vpath1 = os.path.join(video1_dir, vfile1)
        vpath2 = os.path.join(video2_dir, vfile2)

        if not os.path.isfile(vpath1) or not os.path.isfile(vpath2):
            print(f"[WARN] Missing video files: {vfile1}, {vfile2}. Skipped.")
            continue

        len1 = _frame_count(vpath1)
        len2 = _frame_count(vpath2)

        # --- TextGrid の相対パス（sX_uY.TextGrid） ---
        tg_fname = f"{sid}_u{uid}.TextGrid"
        tg_path  = os.path.join(tg_dir, tg_fname)
        if not os.path.isfile(tg_path):
            print(f"[WARN] TextGrid not found: {tg_path}. Skipped.")
            continue

        # --- テキスト読み込み & トークン化 ---
        txt_path = os.path.join(text_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        token_ids = text_transform.tokenize(text).tolist()
        if any((t is None) or (t < 0) for t in token_ids):
            print(f"[NG] {sid}_u{uid}: Invalid token IDs -> {token_ids}. Skipped.")
            continue

        token_str = " ".join(map(str, token_ids))

        # --- 相対パスを 'ディレクトリ名/ファイル名' 形式で出力 ---
        rel_v1 = _rel_path(video1_dir, vfile1)
        rel_v2 = _rel_path(video2_dir, vfile2)
        rel_tg = _rel_path(tg_dir, tg_fname)

        # --- CSV 1行出力 ---
        out.write(f".,{rel_v1},{rel_v2},{rel_tg},{len1},{len2},{token_str}\n")
