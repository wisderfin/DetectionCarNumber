import torch
from utils import SIZE_IMAGE, CHARS_SIZE, CHANELS_IMAGE

img_height, img_weight = SIZE_IMAGE

class CRNN(torch.nn.Module):
    def __init__(
            self,
            img_height: int = img_height,
            num_classes: int = CHARS_SIZE
            ) -> None:
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(CHANELS_IMAGE, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.rnn = torch.nn.LSTM(128 * (img_height // 4), 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x)
        batch_size, channels, height, width = conv.size()

        conv = conv.permute(0, 3, 2, 1).contiguous()
        conv = conv.view(batch_size, width, channels * height) 

        # Подать в LSTM
        rnn_out, _ = self.rnn(conv)
        out = self.fc(rnn_out)
        out = out.permute(1, 0, 2)
        return out
