import csv
import os
import random
import re

# --- 入力 ---
input_csv = "oulu/labels/contrast/all-data.csv"

# --- 出力 ---
train_csv = "oulu/labels/contrast/train.csv"
val_csv   = "oulu/labels/contrast/val.csv"
test_csv  = "oulu/labels/contrast/test.csv"

# --- 話者ごとの割当数（必要に応じて変更） ---
NUM_TEST_SPK = 7
NUM_VAL_SPK  = 7
RAND_SEED    = 42

# --- ユーティリティ ---
SPK_RE = re.compile(r'^(s\d+)_')

def speaker_from_relpath(rel_path: str) -> str | None:
    """rel_path_v1 の 'sX_vY_uZ.mp4' から 'sX' を取り出す"""
    base = os.path.basename(rel_path)
    m = SPK_RE.match(base)
    return m.group(1) if m else None

def same_speaker(rel_path_v1: str, rel_path_v2: str) -> bool:
    s1 = speaker_from_relpath(rel_path_v1)
    s2 = speaker_from_relpath(rel_path_v2)
    return (s1 is not None) and (s1 == s2)

# --- すべての話者IDを抽出 ---
all_speaker_ids = set()
rows = []

with open(input_csv, "r", newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    for row in reader:
        if not row:
            continue  # 空行スキップ
        # 期待形式: . , rel_v1 , rel_v2 , rel_tg , len1 , len2 , token_id
        if len(row) < 7:
            # 列欠損行はスキップ
            print(f"[WARN] Malformed row (len={len(row)}): {row}")
            continue

        dot, rel_v1, rel_v2, rel_tg, len1, len2, token_ids = row[:7]

        spk = speaker_from_relpath(rel_v1)
        if spk is None:
            print(f"[WARN] Could not parse speaker from rel_path_v1: {rel_v1}")
            continue

        # v1/v2 の話者一致チェック（不一致ならスキップ）
        if not same_speaker(rel_v1, rel_v2):
            print(f"[WARN] Speaker mismatch v1 vs v2 -> {rel_v1} | {rel_v2}. Skipped.")
            continue

        all_speaker_ids.add(spk)
        rows.append(row)

all_speaker_ids = sorted(list(all_speaker_ids))

if len(all_speaker_ids) < (NUM_TEST_SPK + NUM_VAL_SPK + 1):
    raise ValueError(
        f"話者が不足しています: total={len(all_speaker_ids)}, "
        f"need >= {NUM_TEST_SPK + NUM_VAL_SPK + 1}"
    )

# --- ランダムに分割（再現性あり） ---
random.seed(RAND_SEED)
random.shuffle(all_speaker_ids)

test_ids = set(all_speaker_ids[:NUM_TEST_SPK])
val_ids  = set(all_speaker_ids[NUM_TEST_SPK:NUM_TEST_SPK+NUM_VAL_SPK])
train_ids = set(all_speaker_ids[NUM_TEST_SPK+NUM_VAL_SPK:])

print(f"[INFO] #speakers total: {len(all_speaker_ids)}")
print(f"[INFO] test_ids: {sorted(test_ids)}")
print(f"[INFO] val_ids : {sorted(val_ids)}")
print(f"[INFO] train_ids: {len(train_ids)} speakers")

# --- 出力準備 ---
os.makedirs(os.path.dirname(train_csv), exist_ok=True)

with open(train_csv, "w", newline="", encoding="utf-8") as train_out, \
     open(val_csv,   "w", newline="", encoding="utf-8") as val_out, \
     open(test_csv,  "w", newline="", encoding="utf-8") as test_out:

    train_writer = csv.writer(train_out)
    val_writer   = csv.writer(val_out)
    test_writer  = csv.writer(test_out)

    for row in rows:
        # row 形式: [".", rel_v1, rel_v2, rel_tg, len1, len2, token_ids]
        rel_v1 = row[1]
        spk = speaker_from_relpath(rel_v1)
        if spk in test_ids:
            test_writer.writerow(row)
        elif spk in val_ids:
            val_writer.writerow(row)
        else:
            train_writer.writerow(row)

print("[DONE] Wrote:", train_csv, val_csv, test_csv)
