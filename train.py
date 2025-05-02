from pathlib import Path
import torch
import torch.utils.data.dataloader
from dataset import ValidationDataset
from crnn import CRNN
from utils import EPOCHS, DEVICE

def train_epoch(dataloader: ValidationDataset) -> float:
    model.train()
    total_loss = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        preds = model(imgs)
        log_probs = torch.nn.functional.log_softmax(preds, dim=2)
        input_lenths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        targets = torch.cat([l for l in labels])
        loss = criterion(log_probs, targets, input_lenths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == '__main__':
    path = Path('src/train')
    train_dataset = ValidationDataset(path)

    def collate_fn(batch):
        images, labels = zip(*batch)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)  # или другой padding_value
        images = torch.stack(images)
        return images, labels_padded

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = CRNN().to(DEVICE)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        loss = train_epoch(train_loader)
        print(f"[{epoch+1}/{EPOCHS}] Loss: {loss:.4f}")
    torch.save(model.state_dict(), "model.pth")
