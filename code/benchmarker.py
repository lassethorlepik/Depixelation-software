import json
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console
from rich.theme import Theme
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Internal tools
from util import check_gpu_compute, print, similarity, decode_label
from dataset_instance import OCRDataset
from model import OCRModel
from gating_network import GatingNetwork


def main():
    print("BENCHMARKER.PY RUNNING ---------------------------------------------")
    device = check_gpu_compute()

    with open("config.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    model_name = params["model_name"]
    dataset_names = params["dataset_names"]
    batch_size = params["batch_size"]
    train_data_amount = params["train_data_amount"]

    with open(Path(f"../datasets/{params['dataset_names'][0]}/metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    img_width = metadata["X"] // metadata["MIN_BLOCK"]
    img_height = metadata["Y"] // metadata["MIN_BLOCK"]

    data_dirs = [Path(f"../datasets/{name}/images") for name in dataset_names]
    dataset = OCRDataset(
        data_dirs=data_dirs,
        img_width=img_width,
        img_height=img_height,
        train_data_amount=train_data_amount,
    )

    test_loader = dataset.get_test_dataloader(
        batch_size=batch_size, shuffle=False
    )

    # Instantiate the model and set it to evaluation mode.
    if params['is_classifier']:
        model = GatingNetwork(
            img_width=img_width,
            img_height=img_height,
            num_to_char=dataset.num_to_char
        ).to(device)
    else:
        model = OCRModel(
            img_width=img_width,
            img_height=img_height,
            num_to_char=dataset.num_to_char,
        ).to(device)
    model.eval()

    # Load the trained weights if they exist.
    checkpoint_path = f"../models/{model_name}/model.pth"
    if os.path.exists(checkpoint_path):
        print("LOADING MODEL...")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print("MODEL LOADED!")
    else:
        print("NO MODEL FOUND!")

    # Initialize benchmarking metrics.
    total_sequences = 0
    total_correct_seqs = 0
    weighted_match_sum = 0
    weighted_total_chars = 0
    match_percentages = []

    # For confusion matrix collection.
    all_true_chars = []
    all_pred_chars = []

    # Set up the rich progress bar.
    custom_theme = Theme({
        "green": "green",
        "light_green": "green1",
        "white": "bold white"
    })
    console = Console(theme=custom_theme)
    progressbar = Progress(
        TextColumn("[progress.description]{task.description}", style="white"),
        BarColumn(bar_width=None, complete_style="green", finished_style="light_green"),
        TaskProgressColumn(style="green"),
        TextColumn("[bold white]Elapsed: "),
        TimeElapsedColumn(),
        TextColumn("[bold white]ETA: "),
        TimeRemainingColumn()
    )

    validation_images = len(test_loader.dataset)
    benchmark_path = f'../models/{model_name}/benchmark'
    os.makedirs(benchmark_path, exist_ok=True)

    with torch.no_grad():
        with progressbar as progress:
            task = progress.add_task(f"Processing 0 / {validation_images}", total=validation_images)
            for batch_num, batch in enumerate(test_loader):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                label_len = batch["label_length"].to(device)

                orig_texts, pred_texts = model.inference_batch(images, labels, label_len)

                n_preds = len(pred_texts)
                n_rows = (n_preds + 3) // 4
                fig, ax = plt.subplots(n_rows, 4, figsize=(10, n_rows * 1.25))
                if n_rows == 1:
                    ax = np.array(ax).reshape(1, -1)
                for i in range(n_preds):
                    pred_text = pred_texts[i]
                    orig_text = orig_texts[i]

                    # Collect confusion matrix data: iterate character by character.
                    for true_char, pred_char in zip(orig_text, pred_text):
                        all_true_chars.append(true_char)
                        all_pred_chars.append(pred_char)

                    match_percentage = similarity(pred_text, orig_text)
                    seq_len = len(orig_text)
                    weighted_match_sum += match_percentage / 100 * seq_len
                    weighted_total_chars += seq_len

                    match_percentages.append(match_percentage)
                    total_sequences += 1
                    total_correct_seqs += (pred_text == orig_text)

                    # Process image for display.
                    img = images[i].cpu().numpy()
                    # If single-channel (shape: [1, H, W]), squeeze to (H, W).
                    if img.shape[0] == 1:
                        img = np.squeeze(img, axis=0)
                    else:
                        # For multichannel images, convert from (C, H, W) to (H, W, C)
                        img = np.transpose(img, (1, 2, 0))

                    # Determine if the prediction exactly matches the ground truth.
                    is_correct = (pred_text == orig_text)
                    color = 'green' if is_correct else 'red'
                    title = f"Pred: {pred_text}\nTrue: {orig_text}\nMatch: {match_percentage:.2f}%"
                    ax[i // 4, i % 4].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                    ax[i // 4, i % 4].set_title(title, fontsize=8, color=color)
                    ax[i // 4, i % 4].axis("off")

                # Hide any unused subplots.
                for j in range(n_preds, n_rows * 4):
                    ax[j // 4, j % 4].axis("off")

                plt.tight_layout()
                plt.savefig(f'{benchmark_path}/batch_{batch_num}.png')
                plt.close()

                progress.update(task, advance=images.size(0),
                                description=f"Processing {total_sequences} / {validation_images}")

    # Compute overall metrics.
    sequence_accuracy_percentage = (total_correct_seqs / total_sequences) * 100 if total_sequences > 0 else 0
    accuracy_percentage = (weighted_match_sum / weighted_total_chars) * 100 if weighted_total_chars > 0 else 0
    sorted_percentages = sorted(match_percentages)
    worst_match_percentage = sorted_percentages[0] if sorted_percentages else 0
    percentile_1st = np.percentile(sorted_percentages, 1) if sorted_percentages else 0

    print(f"BENCHMARK COMPLETE\n")
    print(f"Perfect sequence predictions: {sequence_accuracy_percentage:.2f}%")
    print(f"Predicted characters correctly: {accuracy_percentage:.2f}%")
    print(f"Worst sequence: {worst_match_percentage:.2f}%")
    print(f"1st percentile sequence: {percentile_1st:.2f}%")

    # Save summary to file.
    output_file_path = f'{benchmark_path}/results_summary.txt'
    with open(output_file_path, 'w') as file:
        file.write(f"Perfect sequence predictions: {sequence_accuracy_percentage:.2f}%\n")
        file.write(f"Predicted characters correctly: {accuracy_percentage:.2f}%\n")
        file.write(f"Worst sequence: {worst_match_percentage:.2f}%\n")
        file.write(f"1st percentile sequence: {percentile_1st:.2f}%\n")

        # Compute confusion matrix using the collected character data.
        if all_true_chars and all_pred_chars:
            # Create sorted list of unique labels.
            labels = sorted(list(set(all_true_chars) | set(all_pred_chars)))
            cm = confusion_matrix(all_true_chars, all_pred_chars, labels=labels)

            # Compute maximum off-diagonal value for color scaling.
            mask_diag = np.eye(cm.shape[0], dtype=bool)
            non_diag_values = cm[~mask_diag]
            vmax_non_diag = non_diag_values.max() if non_diag_values.size > 0 else None

            # Plot the confusion matrix heatmap.
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        vmin=0, vmax=vmax_non_diag)
            plt.xlabel("Read symbol")
            plt.ylabel("True symbol")
            plt.title("Confusion matrix")
            plt.tight_layout()
            cm_image_path = f'{benchmark_path}/confusion_matrix.png'
            plt.savefig(cm_image_path)
            plt.close()
            print(f"Confusion matrix image saved to {cm_image_path}")

            # Save the numeric confusion matrix to a text file in a machine-readable CSV format.
            cm_text_path = f'{benchmark_path}/confusion_matrix.txt'
            with open(cm_text_path, 'w') as f:
                for row in cm:
                    f.write(",".join(map(str, row)) + "\n")
            print(f"Confusion matrix data saved to {cm_text_path}")
        else:
            print("No character data collected for confusion matrix.")

if __name__ == "__main__":
    main()
