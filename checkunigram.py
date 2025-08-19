import pandas as pd
import sentencepiece as spm

# 設定
csv_path = "/workspace/auto-avsr/oulu/labels/45/test.csv"  # ← CSVファイルのパス
spm_model_path = "/workspace/auto-avsr/spm/unigram/unigram25.model"  # ← SPMモデルのパス

# SentencePieceモデル読み込み
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

# CSV読み込み（列数が不定なら header=None を指定）
df = pd.read_csv(csv_path, header=None)

# 第3列目（0-indexで3）のトークン列を処理
print("=== REF確認 ===")
for idx, row in df.iterrows():
    token_str = row[3]  # 例："4 15 5 5 16 21 9 3"
    token_ids = list(map(int, token_str.strip().split()))
    decoded = sp.decode(token_ids)

    print(f"[{idx}]")
    print(f" REF token id : {token_ids}")
    print(f" REF decoded  : {decoded}")
    print("-" * 40)
