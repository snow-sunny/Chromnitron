"""Train Chromnitron with user-provided genomic windows and labels.

The script wires a simple PyTorch training loop around the existing
Chromnitron model. It expects:
- genomic sequence and auxiliary feature tracks stored as Zarr
- CAP protein embeddings saved as .npz (key: ``embedding``) or .pt
- a BED file describing training windows used to sample inputs
- target signal/labels stored as a Zarr track aligned to the windows
"""
import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

import chromnitron_data.transforms as transforms
from chromnitron_data.chromnitron_dataset import get_inference_region
from chromnitron_data.origami_infrastructure.storages import ZarrStorage
from chromnitron_data.origami_infrastructure.tracks import Track
from chromnitron_model.chromnitron_models import Chromnitron


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def load_region_bed(path: str):
    loci_info = []
    with open(path, "r") as handle:
        for line in handle:
            chrom, start, end = line.strip().split("\t")[:3]
            loci_info.append([chrom, int(start), int(end)])
    return loci_info


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChromnitronTrainingDataset(Dataset):
    """Minimal dataset for supervised Chromnitron training."""

    def __init__(
        self,
        training_cfg: Dict[str, Any],
        chr_sizes: Dict[str, int],
    ) -> None:
        self.training_cfg = training_cfg
        loci_info = load_region_bed(training_cfg["loci"]) if isinstance(training_cfg["loci"], str) else training_cfg["loci"]
        self.region = get_inference_region(
            loci_info,
            training_cfg["assembly"],
            chr_sizes,
            training_cfg.get("sample_size", 8192),
            training_cfg.get("step_size", 5120),
            training_cfg.get("excluded_region_path"),
        )

        self.seq_track = Track(
            ZarrStorage(training_cfg["sequence_path"], training_cfg["assembly"], chr_sizes)
        )
        self.input_features = self._load_storage_with_paths(
            training_cfg["assembly"], training_cfg.get("input_features_path"), chr_sizes
        )
        self.label_track = Track(
            ZarrStorage(training_cfg["label_path"], training_cfg["assembly"], chr_sizes)
        )
        self.esm_feature = self._load_embeddings(training_cfg["esm_embedding_path"])
        self.use_reverse_aug = training_cfg.get("reverse_aug", False)
        self.use_noise_aug = training_cfg.get("gaussian_noise_aug", False)

    def __len__(self) -> int:
        return len(self.region)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        chrom, start_str, end_str, region_id = self.region[idx]
        start, end = int(start_str), int(end_str)

        seq = transforms.to_onehot(self.seq_track.get(chrom, start, end)).astype("float32")
        input_features = self._get_features(self.input_features, chrom, start, end)
        labels = self.label_track.get(chrom, start, end).astype("float32")

        # Preprocess tracks
        input_features = transforms.log1p_clip_negative(input_features)
        labels = transforms.log1p_clip_negative(labels)

        # Optional data augmentation
        if self.use_reverse_aug:
            seq, input_features, labels = transforms.reverse_features(seq, input_features, labels)
        if self.use_noise_aug:
            seq, input_features, _ = transforms.add_gaussian_noise(seq, input_features, labels)

        seq = seq[None, :, :]
        input_features = input_features[None, :]
        labels = labels[None, :]
        esm_embedding = self.esm_feature[None, :, :]
        return seq, input_features, esm_embedding, labels, (start, end, chrom, region_id)

    def _load_embeddings(self, path: str) -> np.ndarray:
        if path.endswith(".pt"):
            embeddings = torch.load(path)
            if isinstance(embeddings, dict) and "embedding" in embeddings:
                embeddings = embeddings["embedding"]
            embeddings = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        else:
            embeddings = np.load(path)["embedding"]
        return embeddings.astype("float32")

    def _load_storage_with_paths(self, assembly: str, paths: Any, chr_sizes: Dict[str, int]):
        if not paths:
            return None
        if isinstance(paths, str):
            return Track(ZarrStorage(paths, assembly, chr_sizes))
        if isinstance(paths, list):
            return [self._load_storage_with_paths(assembly, p, chr_sizes) for p in paths]
        raise ValueError(f"Invalid paths: {paths}")

    def _get_features(self, features: Any, chrom: str, start: int, end: int):
        if features is None:
            return np.zeros(end - start, dtype="float32")
        if isinstance(features, Track):
            return features.get(chrom, start, end)
        if isinstance(features, list):
            return [self._get_features(feature, chrom, start, end) for feature in features]
        raise ValueError(f"Invalid features: {features}")


def prepare_inputs(seq, input_features):
    batch_size, mini_bs, seq_len, seq_dim = seq.shape
    seq = seq.view(batch_size * mini_bs, seq_len, seq_dim).transpose(1, 2).float()
    input_features = (
        input_features.view(batch_size * mini_bs, -1).unsqueeze(2).transpose(1, 2).float()
    )
    return (seq, input_features)


def train_one_epoch(model, dataloader, optimizer, scaler, device, grad_clip=None):
    model.train()
    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="train"):
        seq, input_features, esm_embeddings, labels, _ = batch
        seq = seq.to(device)
        input_features = input_features.to(device)
        labels = labels.to(device)
        esm_embeddings = esm_embeddings.to(device)
        esm_embeddings = esm_embeddings.float().transpose(-1, -2)

        inputs = prepare_inputs(seq, input_features)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=scaler is not None):
            outputs = model(inputs, esm_embeddings)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = preds.squeeze(1)
            loss = criterion(preds, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
    return running_loss / max(len(dataloader), 1)


def save_checkpoint(model, optimizer, epoch, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )


def load_chr_sizes(path: str):
    chr_sizes = {}
    with open(path, "r") as handle:
        for line in handle:
            chrom, size = line.strip().split("\t")
            chr_sizes[chrom] = int(size)
    return chr_sizes


def main():
    parser = argparse.ArgumentParser(description="Train Chromnitron")
    parser.add_argument("config", type=str, help="Path to training YAML config")
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config.get("seed", 42)
    set_seed(seed)

    training_cfg = config["training_config"]
    data_cfg = config["data_config"]

    chr_sizes = load_chr_sizes(data_cfg["chrom_sizes_path"])
    dataset = ChromnitronTrainingDataset(data_cfg, chr_sizes)
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=True,
    )

    model_cfg = config.get("model_config", {})
    model = Chromnitron(**model_cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
    )

    scaler = torch.cuda.amp.GradScaler() if training_cfg.get("amp", True) and device == "cuda" else None
    grad_clip = training_cfg.get("grad_clip")

    start_epoch = 0
    if training_cfg.get("resume_from"):
        ckpt = torch.load(training_cfg["resume_from"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1

    epochs = training_cfg.get("num_epochs", 1)
    checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints")
    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            grad_clip=grad_clip,
        )
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

        if (epoch + 1) % training_cfg.get("save_every", 1) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    final_ckpt = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, epochs - 1, final_ckpt)


if __name__ == "__main__":
    main()
