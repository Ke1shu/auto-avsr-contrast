import sentencepiece as spm

model_path = "spm/unigram/unigram25.model"
text_path = "spm/input.txt"  # モデル学習に使ったのと同じテキストファイル
output_path = "spm/unigram/unigram25_unitsre.txt"

sp = spm.SentencePieceProcessor()
sp.load(model_path)

# テキストを全て読み込み
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# トークンを1つずつ改行形式にする（スクリプトと同じ）
pieces = sp.encode(text, out_type=str)
unique_pieces = sorted(set(pieces))  # sort + uniq 相当

# 書き出し
with open(output_path, "w", encoding="utf-8") as f:
    f.write("<unk> 1\n")
    for i, token in enumerate(unique_pieces):
        f.write(f"{token} {i + 2}\n")  # <unk>が1なので+2
