import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('spm/unigram/unigram25.model')
text = sp.decode_ids([4, 3, 24, 7, 18, 17, 3, 4, 13, 3])
print(text)

def load_token_list(units_txt_path):
    """units.txt から token_list を構築する"""
    with open(units_txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
        token_list = [line.split()[0] for line in lines]
    return token_list

def decode_ids(token_ids, token_list):
    """token_id のリストから文字列へ変換"""
    tokens = [token_list[i] for i in token_ids]
    return "".join(tokens).replace("▁", " ").strip()

# 使用例
units_path = "spm/unigram/unigram25_units.txt"  # もしくは "unigram25_units.txt"
token_list = load_token_list(units_path)

token_ids = [4, 3, 24, 7, 18, 17, 3, 4, 13, 3]
decoded_text = decode_ids(token_ids, token_list)

print("Decoded text:", decoded_text)
