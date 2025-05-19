import csv
import json
import os
import time
import torch
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gating_network import GatingNetwork
from dataset_instance import OCRDataset
from model import OCRModel
from util import print, check_gpu_compute


def main():
    print("TRAINER.PY RUNNING ---------------------------------------------")
    with open("config.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    device = check_gpu_compute()
    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None

    with open(Path(f"../datasets/{params['dataset_names'][0]}/metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    img_width = metadata["X"] // metadata["MIN_BLOCK"]
    img_height = metadata["Y"] // metadata["MIN_BLOCK"]

    data_dirs = [Path(f"../datasets/{name}/images") for name in params['dataset_names']]
    dataset = OCRDataset(
        data_dirs=data_dirs,
        img_width=img_width,
        img_height=img_height,
        train_data_amount=params['train_data_amount'],
        dataset_limit=params['dataset_limit'],
    )
    dataset.visualize_samples(num_samples=16)
    train_loader = dataset.get_train_dataloader(batch_size=params['batch_size'])
    validation_loader = dataset.get_validation_dataloader(batch_size=params['batch_size'], shuffle=False)

    if params['is_classifier']:
        dataset.model = GatingNetwork(
            img_width=img_width,
            img_height=img_height,
            num_to_char=dataset.num_to_char
        ).to(device)
    else:
        dataset.model = OCRModel(
            img_width=img_width,
            img_height=img_height,
            num_to_char=dataset.num_to_char
        ).to(device)

    optimizer = torch.optim.Adam(dataset.model.parameters(), lr=params['learning_rate'], weight_decay=params['decay'])
    reduce_lr = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params['lr_reduce_factor'],
        patience=params['patience'],
        min_lr=params['min_learning_rate'],
    )

    os.makedirs("../models", exist_ok=True)
    model_folder = f"../models/{params['model_name']}/"
    os.makedirs(model_folder, exist_ok=True)
    metadata_path = os.path.join(model_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    checkpoint_path = os.path.join(model_folder, "model.pth")

    model = dataset.model

    if os.path.exists(checkpoint_path):
        print("Weights found, loading...")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        initial_epoch = checkpoint_data['epoch'] + 1
        model.train_losses = checkpoint_data.get('train_losses', [])
        model.val_losses = checkpoint_data.get('val_losses', [])
        model.cumulative_train_times = checkpoint_data.get('cumulative_train_times', [])
        model.validation_percentages = checkpoint_data.get('validation_percentages', [])
        best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        best_model_state = checkpoint_data['model_state_dict']
        best_optimizer_state = checkpoint_data['optimizer_state_dict']
        print(f"Resuming training from epoch {initial_epoch}")
    else:
        print("No weights found, starting from scratch...")
        initial_epoch = 0
        if params['log_detailed_accuracy']:
            with open(os.path.join(model_folder, "detailed_accuracy.csv"), 'w', newline='') as file:
                pass

    print("Model Summary:")
    print(model)
    os.makedirs(os.path.join(model_folder, "results"), exist_ok=True)
    start_time = time.perf_counter()

    for epoch in range(initial_epoch, params['max_epochs']):
        print(f"Epoch {epoch + 1}/{params['max_epochs']} starting...")

        avg_train_loss = model.train_epoch(train_loader, optimizer, device)
        avg_val_loss, percentages = model.validate(validation_loader, device)

        epoch_duration = time.perf_counter() - start_time
        cumulative_time = model.cumulative_train_times[-1] + epoch_duration if model.cumulative_train_times else epoch_duration
        model.cumulative_train_times.append(cumulative_time)
        start_time = time.perf_counter()

        reduce_lr.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record the losses for later analysis
        model.train_losses.append(avg_train_loss)
        model.val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")
        print(f"LR: {current_lr:.4f} Epoch Duration: {epoch_duration:.2f} sec, Cumulative Time: {cumulative_time / 60.0:.2f} min")

        dataset.visualize_predictions(model, device, epoch, params['model_name'], num_samples=16)

        if params['log_detailed_accuracy']:
            with open(os.path.join(model_folder, "detailed_accuracy.csv"), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(percentages)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            print(f"New best weights found at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'train_losses': model.train_losses,
            'val_losses': model.val_losses,
            'cumulative_train_times': model.cumulative_train_times,
            'validation_percentages': model.validation_percentages,
            'char_to_num': dataset.char_to_num,
            'train_data_amount': params['train_data_amount'],
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_data, checkpoint_path)

    print("Training completed.")


if __name__ == "__main__":
    main()
