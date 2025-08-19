#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import torchaudio
import torchvision


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append((dataset_name, rel_path, int(input_length), torch.tensor([int(_) for _ in token_id.split()])))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}

    def __len__(self):
        return len(self.list)



import re
import textgrid
class ContrastiveAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        # ★ バケツ用の長さは max(frame1, frame2)
        #    self.list の各要素は (dataset_name, rel_path1, rel_path2, textgrid_path, frame1, frame2, token_id)
        self.input_lengths = [max(int(_[4]), int(_[5])) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        """
        CSV 形式:
          root_dir, vid1, vid2, textgrid, frame1, frame2, token
        例:
          oulu,s1_v1.mp4,s1_v2.mp4,s1.TextGrid,120,118,4 19 5 10
          oulu,s2_v1.mp4,s2_v2.mp4,s2.TextGrid,86,86,"86,23,10,5"
        """
        paths_counts_labels = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                # 先頭6カラム＋残り全部(token)を受ける
                # root_dir, vid1, vid2, textgrid, frame1, frame2, token
                parts = line.split(",", 6)
                if len(parts) < 7:
                    raise ValueError(f"Invalid CSV row (need 7 fields): {line}")

                dataset_name   = parts[0].strip()
                rel_path1      = parts[1].strip()
                rel_path2      = parts[2].strip()
                textgrid_path  = parts[3].strip()
                frame1_str     = parts[4].strip()
                frame2_str     = parts[5].strip()
                token_str_raw  = parts[6].strip()

                # フレーム数
                try:
                    frame1 = int(frame1_str)
                    frame2 = int(frame2_str)
                except ValueError:
                    raise ValueError(f"frame1/frame2 must be int: {line}")

                # token は空白/カンマ混在に対応し、空要素は除外。引用符も剥がす
                tok_str = token_str_raw.strip().strip('"').strip("'")
                tok_list = [t for t in re.split(r"[,\s]+", tok_str) if t != ""]
                try:
                    token_id = torch.tensor([int(t) for t in tok_list], dtype=torch.long)
                except ValueError:
                    # 非数が混ざっていた場合に内容を出して気づきやすく
                    raise ValueError(f"token field must be ints separated by space/comma: {token_str_raw}")

                paths_counts_labels.append(
                    (
                        dataset_name,
                        rel_path1,
                        rel_path2,
                        textgrid_path,
                        frame1,
                        frame2,
                        token_id,
                    )
                )
        return paths_counts_labels

    def _load_phoneme(self, path):
        # tg をそのまま返す
        tg = textgrid.TextGrid.fromFile(path)
        return tg

    def __getitem__(self, idx):
        # 7項目に合わせて受ける
        dataset_name, rel_path1, rel_path2, textgrid_path, frame1, frame2, token_id = self.list[idx]
        path1 = os.path.join(self.root_dir, dataset_name, rel_path1)
        path2 = os.path.join(self.root_dir, dataset_name, rel_path2)
        tg_path = os.path.join(self.root_dir, dataset_name, textgrid_path)

        if self.modality == "video":
            video1 = self.video_transform(load_video(path1))
            video2 = self.video_transform(load_video(path2))
            tg = self._load_phoneme(tg_path)

            return {
                "video1": video1,
                "video2": video2,
                "target": token_id,
                "tg": tg,
                # もし下流で参考にしたければ raw frame 情報も返しておける
                "frame1": frame1,
                "frame2": frame2,
            }

        elif self.modality == "audio":
            raise NotImplementedError("2-input audio not supported yet")

    def __len__(self):
        return len(self.list)