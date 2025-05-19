import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def label_to_class_idx(label: str) -> int:
    if label.isdigit():
        return 1
    return 0


class GatingNetwork(nn.Module):
    def __init__(self, img_width: int, img_height: int, num_to_char):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_to_char = num_to_char
        self.num_chars = len(num_to_char)

        self.is_classifier = True
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
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B,256]
            nn.Linear(256 * img_height * img_width, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y_true: torch.Tensor | None = None, label_length: torch.Tensor | None = None):
        """
        x: [B,1,H,W]
        y_true: [B] class indices when training, or None
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.bn(self.conv4(x)))  # [B,256,H,W]

        logits = self.classifier(x)  # [B, num_chars]
        log_probs = F.log_softmax(logits, dim=1)

        if y_true is None:
            return log_probs, None

        loss = F.nll_loss(log_probs, y_true, reduction="none")
        return log_probs, loss

    def decode_batch_predictions(self, log_probs: torch.Tensor, max_length: int, num_to_char: dict[int, str]) -> list[str]:
        """Decode model outputs"""
        texts: list[str] = []
        preds = log_probs.argmax(dim=1)
        for idx in preds.tolist():
            texts.append(num_to_char.get(int(idx), ""))
        return texts

    def train_epoch(self, dataloader, optimizer, device: torch.device) -> float:
        """
        Expects each batch to have:
          - batch['image']: Tensor[B,1,H,W]
          - batch['label']: list of strings

        Returns average epoch loss.
        """
        self.train()
        total_loss = 0.0
        total_images = len(dataloader.dataset)
        bar = tqdm(total=total_images, desc="Train", leave=False)

        for batch in dataloader:
            images = batch["image"].to(device)
            # Convert each string to its class index
            raw_labels = batch["label"].cpu().tolist()
            class_idx_list = []
            for label in raw_labels:
                chars = [self.num_to_char.get(int(value), "") for value in label]
                label_string = "".join(chars)
                class_idx_list.append(label_to_class_idx(label_string))
            labels = torch.tensor(class_idx_list, dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_probs, loss = self(images, labels, None)
            loss_mean = loss.mean()
            loss_mean.backward()
            optimizer.step()

            total_loss += float(loss_mean.detach()) * images.size(0)
            bar.update(images.size(0))
            bar.set_postfix({"img_loss": f"{loss_mean.item():.4f}"})

        bar.close()
        return total_loss / total_images

    def validate(self, dataloader, device):
        self.eval()
        total_loss = 0.0
        all_matches = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                # convert each sequence‐of‐ints label to single class index
                raw_labels = batch["label"].cpu().tolist()
                class_idx_list = []
                for seq in raw_labels:
                    chars = [self.num_to_char.get(int(v), "") for v in seq]
                    label_str = "".join(chars)
                    class_idx = label_to_class_idx(label_str)
                    class_idx_list.append(class_idx)
                labels = torch.tensor(class_idx_list, dtype=torch.long, device=device)  # [B]

                log_probs, loss = self(images, labels, None)
                total_loss += loss.mean().item() * images.size(0)

                # get predicted class per sample
                predictions = log_probs.argmax(dim=1)
                matches = predictions.eq(labels)  # [B] bool tensor
                all_matches.extend(matches.float().tolist())  # list of 0.0/1.0

        avg_val_loss = total_loss / len(dataloader.dataset)
        accuracy = sum(all_matches) / len(all_matches) * 100
        self.validation_percentages.append(accuracy)
        return avg_val_loss, [100 if m else 0 for m in all_matches]

    def inference_batch(self, images, labels=None, label_len=None):
        labels_list = labels.tolist() if labels is not None else []
        class_idx_list = []
        for seq in labels_list:
            chars = [self.num_to_char.get(int(v), "") for v in seq]
            text = "".join(chars)
            class_idx = label_to_class_idx(text)
            class_idx_list.append(class_idx)
        if labels is not None:
            labels = torch.tensor(class_idx_list, dtype=torch.long, device=images.device)  # [B]
        with torch.no_grad():
            log_probs, _ = self(images, labels, label_len)
        orig_texts = [str(i) for i in class_idx_list]
        decoded_texts = [str(i) for i in log_probs.argmax(dim=1).tolist()]
        return orig_texts, decoded_texts
