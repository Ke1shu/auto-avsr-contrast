import torch

ckpt = torch.load("exp/oulu_45_ctc7/epoch=225.ckpt", map_location="cpu")
state_dict = ckpt["state_dict"]

# "model." で始まるキーを除去して保存する
new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
torch.save(new_state_dict, "exp/oulu_45_ctc7/epoch=225.pth")
