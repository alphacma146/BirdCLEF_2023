# Standard lib
from pathlib import Path
import argparse
# Third party
import torch
import torch.nn as nn
import torchaudio  # If Linux it's Sox, if Windows it's SoundFile you need
from torchvision.utils import save_image
from tqdm import tqdm


N_FFT = 2048
HOP_LEN = 1024
BATCH = 100
EPOCH = 10


def create_image(img_path: Path, save_path: Path):
    save_image(
        convert_toimage(img_path),
        save_path
    )


def convert_toimage(img_path: Path) -> torch.Tensor:
    """音声データをtensor型メルスペクトログラムに変換"""
    audio, sample_rate = torchaudio.load(img_path)
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        win_length=HOP_LEN,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        mel_scale="htk"
    )(audio).mean(axis=0)
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    spectrogram = spectrogram - spectrogram.min()
    spectrogram = spectrogram / spectrogram.max()
    return spectrogram


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        drop_rate,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            growth_rate * 4,
            kernel_size=1,
            bias=False
        )
        self.norm2 = nn.BatchNorm2d(growth_rate * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            growth_rate * 4,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x


class DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers,
        in_channels,
        growth_rate,
        drop_rate,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, x0):
        x = [x0]
        for name, layer in self.items():
            out = layer(x)
            x.append(out)

        return torch.cat(x, 1)


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate,
        block_config,
        drop_rate=0,
        num_classes=1000,
    ):
        super().__init__()

        # 最初の畳み込み層を追加する。
        self.features = nn.Sequential()
        self.features.add_module(
            "conv0",
            nn.Conv2d(
                3,
                2 * growth_rate,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(2 * growth_rate))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module(
            "pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block 及び Transition Layer を作成する。
        in_channels = 2 * growth_rate
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=in_channels,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)

            in_channels = in_channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                # 最後の Dense Block でない場合は、Transition Layer を追加する。
                trans = TransitionLayer(
                    in_channels=in_channels,
                    out_channels=in_channels // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = in_channels // 2

        self.features.add_module("norm5", nn.BatchNorm2d(in_channels))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.features.add_module("pool5", nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(in_channels, num_classes)

        # 重みを初期化する。
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def densenet121():
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16))


def densenet169():
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32))


def densenet201():
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32))


def densenet264():
    return DenseNet(growth_rate=48, block_config=(6, 12, 64, 48))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        action="store_true",
        help="Create melspectrogram image"
    )
    args = parser.parse_args()

    output_folder = Path("Data\\melspectrogram")
    if args.image:
        input_folder = Path("Data\\train_audio")
        for data_path in tqdm(input_folder.rglob("*.ogg")):
            output_path = output_folder / data_path.parent.name
            if not output_path.exists():
                output_path.mkdir(parents=True)
            create_image(data_path, output_path / (data_path.stem + ".png"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = densenet169().to(device)
    print(type(model))
