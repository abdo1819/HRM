from typing import List, Dict
import os
import json

import numpy as np
from tqdm import tqdm

import torchaudio
from transformers import WhisperProcessor

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    root: str = "dataset/raw-data/LibriSpeech"
    output_dir: str = "data/librispeech"

    train_subsets: List[str] = ["train-clean-100"]
    valid_subsets: List[str] = ["validation-clean"]
    test_subsets: List[str] = ["test-clean"]

    processor_name: str = "openai/whisper-tiny"


def _encode_transcripts(texts: List[str]) -> Dict[str, np.ndarray]:
    """Build character mapping and encode transcripts to padded arrays."""
    char_set = set()
    max_len = 0
    for t in texts:
        char_set.update(t)
        if len(t) > max_len:
            max_len = len(t)

    char_list = sorted(char_set)
    char2id = {c: i + 1 for i, c in enumerate(char_list)}  # 0 reserved for PAD

    labels = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            labels[i, j] = char2id[c]

    return {
        "labels": labels,
        "vocab_size": len(char2id) + 1,
        "seq_len": max_len,
    }


def _convert_split(split_name: str, urls: List[str], processor: WhisperProcessor, config: DataProcessConfig):
    save_dir = os.path.join(config.output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    features: List[np.ndarray] = []
    transcripts: List[str] = []

    for url in urls:
        dataset = torchaudio.datasets.LIBRISPEECH(config.root, url=url, download=True)
        for waveform, sample_rate, transcript, *_ in tqdm(dataset, desc=f"{split_name}:{url}"):
            waveform = waveform.mean(dim=0)
            if sample_rate != processor.feature_extractor.sampling_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, processor.feature_extractor.sampling_rate
                )
                sample_rate = processor.feature_extractor.sampling_rate

            feats = processor.feature_extractor(
                waveform.numpy(), sampling_rate=sample_rate, return_tensors="np"
            ).input_features[0]
            features.append(feats)
            transcripts.append(transcript)

    if not features:
        return

    label_info = _encode_transcripts(transcripts)
    inputs = np.stack(features)

    # indices for PuzzleDataset format (each sample is its own puzzle/group)
    num_examples = inputs.shape[0]
    puzzle_identifiers = np.zeros(num_examples, dtype=np.int32)
    puzzle_indices = np.arange(num_examples + 1, dtype=np.int32)
    group_indices = np.arange(num_examples + 1, dtype=np.int32)

    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), label_info["labels"])
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)

    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=label_info["vocab_size"],
        seq_len=label_info["seq_len"],
        num_puzzle_identifiers=1,
        total_groups=num_examples,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    processor = WhisperProcessor.from_pretrained(config.processor_name)

    splits = {
        "train": config.train_subsets,
        "validation": config.valid_subsets,
        "test": config.test_subsets,
    }

    for split_name, urls in splits.items():
        if urls:
            _convert_split(split_name, urls, processor, config)


if __name__ == "__main__":
    cli()
