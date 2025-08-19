import csv
import random

# --- 入力 ---
input_csv = "oulu/labels/45/all-data.csv"

# --- 出力 ---
train_csv = "oulu/labels/45/train.csv"
val_csv = "oulu/labels/45/val.csv"
test_csv = "oulu/labels/45/test.csv"

# --- すべての話者IDを抽出 ---
all_speaker_ids = set()
with open(input_csv, 'r') as infile:
    for line in infile:
        filename = line.split(',')[1]
        basename = filename.split('/')[-1]
        speaker_id = basename.split('_')[0]  # 例: s1
        all_speaker_ids.add(speaker_id)

# --- ランダムに分割 ---
all_speaker_ids = list(all_speaker_ids)
random.seed(42)  # 再現性のあるランダム
random.shuffle(all_speaker_ids)

test_ids = set(all_speaker_ids[:7])
val_ids = set(all_speaker_ids[7:14])
train_ids = set(all_speaker_ids[14:])

print(f"[INFO] test_ids: {sorted(test_ids)}")
print(f"[INFO] val_ids : {sorted(val_ids)}")
print(f"[INFO] train_ids: {len(train_ids)} speakers")

# --- 振り分け処理 ---
with open(input_csv, 'r') as infile, \
     open(train_csv, 'w') as train_out, \
     open(val_csv, 'w') as val_out, \
     open(test_csv, 'w') as test_out:

    for line in infile:
        filename = line.split(',')[1]
        basename = filename.split('/')[-1]
        speaker_id = basename.split('_')[0]

        if speaker_id in test_ids:
            test_out.write(line)
        elif speaker_id in val_ids:
            val_out.write(line)
        else:
            train_out.write(line)
