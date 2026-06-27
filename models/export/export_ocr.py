"""
Export lightweight OCR model to ExecuTorch .pte for Snapdragon NPU.

This exports a CRNN-based text recognition model for reading serial numbers
and asset tags from server rack labels.

Usage:
    python models/export/export_ocr.py --soc_model SM8750 --compile_only
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class LightweightCRNN(nn.Module):
    """Minimal CRNN for alphanumeric serial number recognition.

    Input: (1, 1, 32, 128) grayscale cropped label region
    Output: (seq_len, batch, num_classes) character predictions
    """

    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_. "
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=128 * 4, hidden_size=128, num_layers=1, bidirectional=True
        )

        self.fc = nn.Linear(256, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x)  # (B, 128, 4, W/4)
        b, c, h, w = conv.size()
        conv = conv.permute(3, 0, 1, 2).reshape(w, b, c * h)  # (W/4, B, 128*4)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)  # (W/4, B, NUM_CLASSES)
        return output


def export_to_pte(model, example_input, output_path: Path, soc_model: str):
    """Export PyTorch model → ExecuTorch .pte via QNN backend."""
    from executorch.backends.qualcomm.partition import QnnPartitioner
    from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED
    from executorch.exir import to_edge

    exported = torch.export.export(model, (example_input,))
    edge_program = to_edge(exported)

    partitioner = QnnPartitioner(
        {
            "soc_model": soc_model,
            QCOM_QUANTIZED: True,
        }
    )
    edge_program = edge_program.to_backend(partitioner)
    et_program = edge_program.to_executorch()

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export OCR model for ExecuTorch")
    parser.add_argument("--soc_model", default="SM8750")
    parser.add_argument("--output", default="models/dc_ops_ocr.pte")
    parser.add_argument("--compile_only", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = LightweightCRNN().eval()
    example_input = torch.randn(1, 1, 32, 128)

    print(f"CRNN params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Exporting for {args.soc_model}...")

    export_to_pte(model, example_input, output_path, args.soc_model)
    print("Done.")


if __name__ == "__main__":
    main()
