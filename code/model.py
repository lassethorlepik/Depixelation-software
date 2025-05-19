from itertools import groupby
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from util import decode_label, similarity


def ctc_decode(y_pred: torch.Tensor, blank: int = 0):
    """Greedy CTC decode (collapse repeats + remove blanks)"""
    # [T, B, C] to [B, T]
    max_idxs = y_pred.argmax(dim=2).transpose(0, 1)
    decoded = []
    for seq in max_idxs.tolist():
        out = [k for k, _ in groupby(seq) if k != blank]
        decoded.append(torch.as_tensor(out, dtype=torch.long, device=y_pred.device))
    return decoded, None


class OCRModel(nn.Module):
    def __init__(self, img_width: int, img_height: int, num_to_char):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_to_char = num_to_char
        self.num_chars = len(num_to_char)

        self.is_classifier = False
        self.train_losses = []
        self.val_losses = []
        self.cumulative_train_times = []
        self.validation_percentages = []

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        lstm_input_size = img_height * 256
        self.lstm0 = nn.LSTM(lstm_input_size, 512, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm1 = nn.LSTM(2 * 512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(2 * 256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)

        self.fc = nn.Linear(2 * 128, self.num_chars)  # final classifier, includes blank

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y_true: torch.Tensor | None = None, label_length: torch.Tensor | None = None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.bn(x)
        x = self.relu(x)  # [B, 256, H, W]

        # Prepare for RNN
        x = x.permute(0, 3, 2, 1)  # [B, W, H, 256]
        B, W, H, C = x.size()
        x = x.reshape(B, W, H * C)  # [B, W, H*256]

        x, _ = self.lstm0(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc(x)  # [B, W, num_chars]
        log_probs = F.log_softmax(x, dim=2).permute(1, 0, 2)  # [T=W, B, C]
        T = log_probs.size(0)

        if y_true is None:
            return log_probs, None

        if label_length is None:
            raise ValueError("label_length must be provided when y_true is not None")

        input_length = torch.full((B,), T, dtype=torch.long, device=x.device)
        loss = F.ctc_loss(
            log_probs,
            y_true,
            input_length,
            label_length,
            blank=0,
            reduction="none",
            zero_infinity=True,
        )
        return log_probs, loss

    def train_epoch(self, dataloader, optimizer, device: torch.device):
        """One training epoch. The dataloader must yield dicts with keys:
            image Tensor[B, 1, H, W]
            label Tensor[B, max_label_len] (padded)
            label_len Tensor[B] (non‑pad counts)

        Returns Average epoch loss (mean of per‑image losses).
        """
        self.train()
        total_images = len(dataloader.dataset)
        bar = tqdm(total=total_images, desc="Train", leave=False)

        running_loss = 0.0
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            label_len = batch["label_length"].to(device)

            optimizer.zero_grad()
            _, loss = self(images, labels, label_len)
            loss_mean = loss.mean()
            loss_mean.backward()
            optimizer.step()

            running_loss += float(loss_mean.detach())
            bar.update(images.size(0))
            bar.set_postfix({"img_loss": f"{loss_mean.item():.4f}"})

        bar.close()
        return running_loss / len(dataloader)

    def decode_batch_predictions(self, log_probs: torch.Tensor, max_length: int, num_to_char: dict[int, str]):
        """Convert network log‑probs to a list of decoded strings."""
        decoded, _ = ctc_decode(log_probs)  # list of length B
        texts: list[str] = []
        for seq in decoded:
            if max_length is not None:
                seq = seq[:max_length]
            chars = [num_to_char.get(int(idx), "") for idx in seq]
            texts.append("".join(chars))
        return texts

    def validate(self, dataloader, device):
        self.eval()
        total_loss = 0.0
        percentages = []
        weighted_match_sum = 0.0
        weighted_total_chars = 0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                label_len = batch["label_length"].to(device)
                # Forward pass with loss computation.
                log_probs, loss = self(images, labels, label_len)
                total_loss += loss.mean().item()

                decoded_texts = self.decode_batch_predictions(log_probs, 20, self.num_to_char)
                for i, text in enumerate(decoded_texts):
                    decoded_label = decode_label(self.num_to_char, labels[i])
                    match_percentage = similarity(text, decoded_label)
                    percentages.append(match_percentage)
                    seq_len = len(decoded_label)
                    weighted_match_sum += match_percentage * seq_len
                    weighted_total_chars += seq_len

        if weighted_total_chars > 0:
            avg_percentage = weighted_match_sum / weighted_total_chars
        else:
            avg_percentage = 0
        self.validation_percentages.append(avg_percentage)
        avg_val_loss = total_loss / len(dataloader)
        return avg_val_loss, percentages

    def inference_batch(self, images, labels=None, label_len=None):
        with torch.no_grad():
            log_probs, _ = self(images, labels, label_len)
        orig_texts = [decode_label(self.num_to_char, labels[i]) for i in range(len(labels if labels is not None else []))]
        decoded_texts = self.decode_batch_predictions(log_probs, 20, self.num_to_char)
        return orig_texts, decoded_texts
