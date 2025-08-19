import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule

import torch.nn.functional as F


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

        # -- initialise
        if getattr(args, "pretrained_model_path", None):
            ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            if getattr(args, "transfer_frontend", False):
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend component are loaded successfully.")
            elif getattr(args, "transfer_encoder", False):
                tmp_ckpt = {k.replace("frontend.",""):v for k,v in ckpt.items() if k.startswith("frontend.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("proj_encoder.",""):v for k,v in ckpt.items() if k.startswith("proj_encoder.")}
                self.model.proj_encoder.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("encoder.",""):v for k,v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
            else:
                self.model.load_state_dict(ckpt)
                print("Pretrained weights of the full model are loaded successfully.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        x = self.model.frontend(sample.unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    '''
    def test_step(self, sample, sample_idx):
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return
    '''
    def test_step(self, sample, sample_idx):
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]

        if not nbest_hyps:
            print(f"[DEBUG] Sample {sample_idx}: beam_search returned EMPTY")
            predicted = ""
            yseq = []
        else:
            yseq = nbest_hyps[0]["yseq"][1:]  # <sos> を除去
            predicted_token_id = torch.tensor(list(map(int, yseq)))
            predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        print(f"[DEBUG] Sample {sample_idx}")
        print(f" > REF (decoded): '{actual}'")
        print(f" > HYP (decoded): '{predicted}'")
        print(f" > REF token id: {actual_token_id.tolist()}")
        print(f" > HYP token id: {yseq}")
        print(f" > target type    : {type(sample['target'])}")

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch["inputs"].size(0)
        batch_sizes = self.all_gather(batch_size)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.7,
    lm_weight=0.0,
    beam_size=10,#ここの数値なににしたらいいかわからん
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


import textgrid

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

@torch.no_grad()
def _build_phoneme_dict_batch_perview(tgs, len1, len2, enc_len1, enc_len2, fps, tier="phones"):
    """
    tgs: List[textgrid.TextGrid], 長さB
    len1,len2: [B]  (入力映像フレーム長; Transform 後)
    enc_len1,enc_len2: [B] (encoder 出力の実長)
    return:
      { phoneme: {
          "v1": [ [(s,e),...], [(s,e),...], ... ],  # 長さB
          "v2": [ [(s,e),...], [(s,e),...], ... ],
        }, ... }
    """
    B = len(tgs)
    out = {}
    for b, tg in enumerate(tgs):
        phones = next(t for t in tg.tiers if t.name == tier)
        Tin1, Tenc1 = int(len1[b]), int(enc_len1[b])
        Tin2, Tenc2 = int(len2[b]), int(enc_len2[b])
        r1 = Tenc1 / max(1, Tin1)
        r2 = Tenc2 / max(1, Tin2)
        for it in phones.intervals:
            ph = (it.mark or "").strip()
            if not ph:
                continue
            s_in = int(round(it.minTime * fps))
            e_in = int(round(it.maxTime * fps))
            s1 = max(0, min(Tenc1 - 1, int(round(s_in * r1))))
            e1 = max(0, min(Tenc1 - 1, int(round(e_in * r1))))
            if e1 < s1: s1, e1 = e1, s1
            s2 = max(0, min(Tenc2 - 1, int(round(s_in * r2))))
            e2 = max(0, min(Tenc2 - 1, int(round(e_in * r2))))
            if e2 < s2: s2, e2 = e2, s2
            if ph not in out:
                out[ph] = {"v1": [[] for _ in range(B)], "v2": [[] for _ in range(B)]}
            out[ph]["v1"][b].append((s1, e1))
            out[ph]["v2"][b].append((s2, e2))
    return out

def _pad_to_same_T(z1, l1, z2, l2):
    """
    z1: [B,T1,D], l1:[B], z2:[B,T2,D], l2:[B] -> 同じ T へ右パディング
    """
    B, T1, D = z1.shape
    T2 = z2.shape[1]
    Tm = max(T1, T2)
    if T1 < Tm:
        z1 = torch.cat([z1, z1.new_zeros(B, Tm - T1, D)], dim=1)
    if T2 < Tm:
        z2 = torch.cat([z2, z2.new_zeros(B, Tm - T2, D)], dim=1)
    return z1, l1, z2, l2, Tm

def contrastive_phonem_loss_Tmasked_perview(z1, z2, phoneme_dict, len1, len2, temperature=0.1):
    """
    z1,z2: [B,T,D] (同じTに右パディング済)
    len1,len2: [B] 実長（pad は損失から除外）
    phoneme_dict: _build_phoneme_dict_batch_perview の戻り

    - 同一バッチ内の「同じ音素」の区間内フレーム同士が正例
      (v1-v1, v1-v2, v2-v1, v2-v2 の4通り)
    - パディング行/列は sim/mask から完全除外
    """
    device = z1.device
    z1 = z1.permute(1, 0, 2).contiguous()  # [T,B,D]
    z2 = z2.permute(1, 0, 2).contiguous()  # [T,B,D]
    T, B, D = z1.shape

    z1 = F.normalize(z1, dim=2)
    z2 = F.normalize(z2, dim=2)

    z1_flat = z1.reshape(-1, D)  # [T*B, D]
    z2_flat = z2.reshape(-1, D)  # [T*B, D]
    representation = torch.cat([z1_flat, z2_flat], dim=0)  # [2TB, D]
    sim = torch.matmul(representation, representation.t())  # [2TB, 2TB]

    def idx_v1(t, b): return t * B + b
    def idx_v2(t, b): return T * B + (t * B + b)

    # 正例マスク（バッチ横断 & ビュー内/跨ぎ）
    mask = torch.zeros_like(sim, dtype=torch.bool, device=device)
    for _, view_lists in phoneme_dict.items():
        v1_lists = view_lists["v1"]
        v2_lists = view_lists["v2"]
        for b1 in range(B):
            r1_v1 = v1_lists[b1]; r1_v2 = v2_lists[b1]
            if not (r1_v1 or r1_v2): continue
            for b2 in range(B):
                r2_v1 = v1_lists[b2]; r2_v2 = v2_lists[b2]
                if not (r2_v1 or r2_v2): continue
                # v1-v1
                for (s1, e1) in r1_v1:
                    for (s2, e2) in r2_v1:
                        ts1 = torch.arange(s1, e1 + 1, device=device)
                        ts2 = torch.arange(s2, e2 + 1, device=device)
                        mask[idx_v1(ts1[:, None], b1), idx_v1(ts2[None, :], b2)] = True
                # v1-v2
                for (s1, e1) in r1_v1:
                    for (s2, e2) in r2_v2:
                        ts1 = torch.arange(s1, e1 + 1, device=device)
                        ts2 = torch.arange(s2, e2 + 1, device=device)
                        mask[idx_v1(ts1[:, None], b1), idx_v2(ts2[None, :], b2)] = True
                # v2-v1
                for (s1, e1) in r1_v2:
                    for (s2, e2) in r2_v1:
                        ts1 = torch.arange(s1, e1 + 1, device=device)
                        ts2 = torch.arange(s2, e2 + 1, device=device)
                        mask[idx_v2(ts1[:, None], b1), idx_v1(ts2[None, :], b2)] = True
                # v2-v2
                for (s1, e1) in r1_v2:
                    for (s2, e2) in r2_v2:
                        ts1 = torch.arange(s1, e1 + 1, device=device)
                        ts2 = torch.arange(s2, e2 + 1, device=device)
                        mask[idx_v2(ts1[:, None], b1), idx_v2(ts2[None, :], b2)] = True

    # パディング完全除外（行列の行/列から落とす）
    valid = torch.zeros(2 * T * B, dtype=torch.bool, device=device)
    for b in range(B):
        L1 = int(len1[b])
        if L1 > 0:
            valid[(torch.arange(L1, device=device) * B + b)] = True
        L2 = int(len2[b])
        if L2 > 0:
            valid[T * B + (torch.arange(L2, device=device) * B + b)] = True

    sim = sim[valid][:, valid]
    mask = mask[valid][:, valid]

    # InfoNCE（行毎の分母）
    positive_exp = torch.exp(sim[mask] / temperature)
    denominator = torch.exp(sim / temperature).sum(dim=1, keepdim=True)
    row_idx = mask.nonzero(as_tuple=False)[:, 0]
    denom_for_pos = denominator[row_idx, 0]
    all_losses = -torch.log(positive_exp / denom_for_pos)
    loss = all_losses.mean() if all_losses.numel() > 0 else sim.new_tensor(0.0)
    return loss

class ContrastiveModelModule(LightningModule):
    """
    既存の ModelModule には触れず、対照学習用の別モジュールを追加。
    ESPnet の E2E を既存と同様に使い、encoder 出力を取り出して損失にかける。
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # 既存 lightning.py と同じ流儀でトークン辞書を用意
        from datamodule.transforms import TextTransform
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        # 既存と同じ E2E を作成（前処理/encoder/ctc/decoder を内包）
        self.model = E2E(len(self.token_list), args.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

        # 既存コードに合わせて任意の事前学習のロードがあればここで（省略可）

        # pre-contrastive 設定
        self.fps = getattr(args, "fps", 25.0)
        self.tg_tier = getattr(args, "tg_tier", "phones")
        self.alpha = getattr(args, "contrastive_weight", 0.5)

        self.lr = getattr(args, "lr", 1e-3)
        self.weight_decay = getattr(args, "weight_decay", 1e-4)

        # 既存の余熱付きCosineがあればそれを使う（cosine.WarmupCosineScheduler）
        from cosine import WarmupCosineScheduler
        self._scheduler_class = WarmupCosineScheduler

    # ===== E2E の encoder を直叩きして時系列表現を取得 =====
    def _forward_encoder(self, xs, ilens):
        """
        xs: [B,T,1,H,W]（video）/ [B,T,feat]（audio）
        戻り: hs_pad [B,Tenc,D], hs_len [B]
        """
        lengths = ilens.clone()
        if self.args.modality == "audio":
            # 既存 E2E.forward と同じ仕様（音声は 640 でダウンサンプル）
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(lengths).to(xs.device).unsqueeze(-2)  # [B,1,T]
        x = self.model.frontend(xs)           # ResNet(frontend)
        x = self.model.proj_encoder(x)        # 次元合わせ
        x, _ = self.model.encoder(x, padding_mask)  # ConformerEncoder
        hs_pad = x
        hs_len = lengths
        return hs_pad, hs_len

    def _ctc_att_loss(self, xs, ilens, ys):
        """
        E2E.forward(x, lengths, label) は (loss, loss_ctc, loss_att, acc) を返す。
        合成損失 loss を使う。
        """
        loss, _, _, _ = self.model(xs, ilens, ys)
        return loss

    # ===== Lightning 標準 =====
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # バッチ（collate_pad の戻り）を受ける
        xs1, len1 = batch["inputs1"], batch["input_lengths1"]  # [B,T1max,1,H,W], [B]
        xs2, len2 = batch["inputs2"], batch["input_lengths2"]  # [B,T2max,1,H,W], [B]
        ys = batch["targets"]                                  # [B,Lmax]
        tgs = batch["tgs"]                                     # List[TextGrid], 長さB

        # 1) Encoder 出力（ビュー別）
        enc1, enc_len1 = self._forward_encoder(xs1, len1)  # [B,Tenc1,D], [B]
        enc2, enc_len2 = self._forward_encoder(xs2, len2)  # [B,Tenc2,D], [B]

        # 2) T を右パディングで統一（pad は後で損失から除外）
        z1, l1, z2, l2, _ = _pad_to_same_T(enc1, enc_len1, enc2, enc_len2)  # [B,T,D], ...

        # 3) TextGrid -> phoneme_dict（バッチ版/ビュー別）
        phoneme_dict = _build_phoneme_dict_batch_perview(
            tgs, len1, len2, enc_len1, enc_len2, fps=self.fps, tier=self.tg_tier
        )

        # 4) 対照損失（pad を分子/分母から完全除外、同音素=正例／バッチ横断）
        loss_contrastive = contrastive_phonem_loss_Tmasked_perview(
            z1, z2, phoneme_dict, l1, l2, temperature=0.1
        )



        # 5) CTC+Attention 損失（各ビュー）
        loss_ctcatt_v1 = self._ctc_att_loss(xs1, len1, ys)
        loss_ctcatt_v2 = self._ctc_att_loss(xs2, len2, ys)

        # 6) 合算
        loss = 0.5 * (loss_ctcatt_v1 + loss_ctcatt_v2) + self.alpha * loss_contrastive

        self.log_dict({
            "loss": loss,
            "loss_contrastive": loss_contrastive,
            "loss_ctcatt_v1": loss_ctcatt_v1,
            "loss_ctcatt_v2": loss_ctcatt_v2,
        }, on_step=True, on_epoch=True, prog_bar=True, batch_size=xs1.size(0))
        return loss


    def validation_step(self, batch, batch_idx):
        xs1, len1 = batch["inputs1"], batch["input_lengths1"]
        xs2, len2 = batch["inputs2"], batch["input_lengths2"]
        ys = batch["targets"]
        tgs = batch["tgs"]

        # 1) Encoder 出力
        enc1, enc_len1 = self._forward_encoder(xs1, len1)
        enc2, enc_len2 = self._forward_encoder(xs2, len2)

        # 2) 同じ T へ右パディング
        z1, l1, z2, l2, _ = _pad_to_same_T(enc1, enc_len1, enc2, enc_len2)

        # 3) TextGrid -> phoneme_dict
        phoneme_dict = _build_phoneme_dict_batch_perview(
            tgs, len1, len2, enc_len1, enc_len2, fps=self.fps, tier=self.tg_tier
        )

        # 4) 対照損失（val）
        loss_contrastive = contrastive_phonem_loss_Tmasked_perview(
            z1, z2, phoneme_dict, l1, l2, temperature=0.1
        )
        self.log(
            "loss_contrastive_val",
            loss_contrastive,
            on_step=False, on_epoch=True,
            batch_size=xs1.size(0),
            sync_dist=True,
        )

        # 5) CTC+Attention（各ビュー）
        loss_ctcatt_v1 = self._ctc_att_loss(xs1, len1, ys)
        loss_ctcatt_v2 = self._ctc_att_loss(xs2, len2, ys)

        # 6) 合算（valの総合lossも見たい場合）
        loss = 0.5 * (loss_ctcatt_v1 + loss_ctcatt_v2) + self.alpha * loss_contrastive
        self.log(
            "loss_val",
            loss,
            on_step=False, on_epoch=True,
            batch_size=xs1.size(0),
            sync_dist=True,
        )
        return loss

