import json
import math
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console import Console
from rich.theme import Theme

# Internal tools and modules
from util import print, decode_label, check_gpu_compute
from dataset_instance import OCRDataset
from model import OCRModel, decode_batch_predictions


def display_inference_results(output_dir, cols=5):
    """Display saved inference images in a grid using matplotlib."""
    image_paths = sorted(output_dir.glob('*.png'))
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found in the output directory.")
        return
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 6, rows * 6))
    for idx, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        plt.subplot(rows, cols, idx + 1)
        if img.ndim == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    print("INFERENCE SCRIPT RUNNING ---------------------------------------------")
    device = check_gpu_compute()
    folder_path = Path("../inference_input")
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions]
    if not image_paths:
        print("No images found in the selected folder.")
        return

    with open("config.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    checkpoint_path = Path(f"../models/{params['model_name']}/model.pth")
    if checkpoint_path.exists():
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        char_to_num = checkpoint_data.get('char_to_num', [])
        charset = ''.join(char_to_num.keys())
        print("MODEL LOADED!")
    else:
        print("NO MODEL FOUND!")
        return

    # Get image dimensions from the first image.
    with Image.open(image_paths[0]) as img:
        img_width, img_height = img.size

    batch_size = 4
    train_data_amount = 0

    dataset = OCRDataset(
        data_dirs=[folder_path],
        img_width=img_width,
        img_height=img_height,
        train_data_amount=train_data_amount,
        charset=charset
    )

    validation_loader = dataset.get_validation_dataloader(
        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    model = OCRModel(
        img_width=img_width,
        img_height=img_height,
        num_chars=len(dataset.num_to_char)
    ).to(device)
    model.eval()

    model.load_state_dict(checkpoint_data['model_state_dict'])

    output_dir = Path('../inference_results')
    output_dir.mkdir(exist_ok=True)

    # Set up the progress bar.
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

    total_images = len(validation_loader.dataset)

    with torch.no_grad():
        with progressbar as progress:
            task = progress.add_task(f"Processing 0 / {total_images}", total=total_images)
            for batch_num, batch in enumerate(validation_loader):
                images = batch['image'].to(device)
                labels = batch.get('label', None)
                preds, loss = model(images)
                pred_texts = decode_batch_predictions(preds, dataset.max_length, dataset.num_to_char)
                orig_texts = []
                for label in labels:
                    orig_texts.append(decode_label(dataset.num_to_char, label))

                for i in range(len(pred_texts)):
                    pred_text = pred_texts[i]
                    # Convert image tensor to numpy array
                    img = images[i].cpu().numpy()
                    # If single-channel (shape: [1, H, W]), squeeze to (H, W)
                    if img.shape[0] == 1:
                        img = np.squeeze(img, axis=0)
                    else:
                        # Convert from (C, H, W) to (H, W, C)
                        img = np.transpose(img, (1, 2, 0))

                    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
                    ax.imshow(img, cmap="gray")
                    ax.set_title(pred_text, fontsize=10, color="black", pad=5)
                    ax.axis("off")

                    save_filename = f"{orig_texts[i]}.png"
                    save_path = output_dir / save_filename
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                progress.update(task, advance=images.size(0), description=f"Processing {batch_num + 1} / {total_images}")

    print("Inference completed.")
    display_inference_results(output_dir)


if __name__ == "__main__":
    main()
