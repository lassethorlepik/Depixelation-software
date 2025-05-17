import os
import sys
import torch
from itertools import zip_longest

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))


def select_model():
    """ List available models under ../models and prompt to choose one."""
    try:
        model_dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    except FileNotFoundError:
        print(f"Models directory not found: {MODELS_DIR}")
        sys.exit(1)

    if not model_dirs:
        print(f"No model folders found in: {MODELS_DIR}")
        sys.exit(1)

    print("Available models:")
    for idx, name in enumerate(model_dirs, start=1):
        print(f" [{idx}] {name}")

    while True:
        choice = input("Select model index: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(model_dirs):
            sel = model_dirs[int(choice) - 1]
            model_path = os.path.join(MODELS_DIR, sel, 'model.pth')
            if os.path.exists(model_path):
                return model_path
            else:
                print(f"Error: 'model.pth' not found in folder '{sel}'.")
        else:
            print("Invalid choice. Enter a number from the list.")


def get_model_stats(filepath, show=100):
    """Loads the checkpoint and returns a string with model stats"""
    try:
        ckpt = torch.load(filepath, map_location="cpu")
    except Exception as e:
        return f"Error loading checkpoint: {e}"

    trains = ckpt.get('train_losses', [])
    vals = ckpt.get('val_losses', [])
    cumulative = ckpt.get('cumulative_train_times', [])
    accuracies = ckpt.get('validation_percentages', [])

    lines = []
    durations = [cumulative[i] - (cumulative[i - 1] if i else 0) for i in range(len(cumulative))]
    data_chunks = [
        ('Train loss', trains[-show:]),
        ('Val loss', vals[-show:]),
        ('Val CRR (%)', accuracies[-show:]),
        ('Cumulative (min)', [c / 60 for c in cumulative[-show:]]),
        ('Duration (min)', [d / 60 for d in durations[-show:]])
    ]

    rows = []
    start_epoch = max(1, len(trains) - len(data_chunks[0][1]) + 1)
    for idx, vals_row in enumerate(zip_longest(*(chunk[1] for chunk in data_chunks), fillvalue=None)):
        epoch_num = start_epoch + idx
        formatted = []
        for (col, _), val in zip(data_chunks, vals_row):
            if val is None:
                cell = "N/A"
            elif 'Loss' in col:
                cell = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            elif 'Acc' in col:
                cell = f"{val:.2f}%" if isinstance(val, (int, float)) else str(val)
            else:
                cell = f"{float(val):.2f}" if isinstance(val, (int, float)) else str(val)
            formatted.append(cell)
        rows.append([str(epoch_num)] + formatted)

    headers = ['Epoch'] + [col for col, _ in data_chunks]
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(item) for item in col) for col in cols]
    lines.append(f"Last {show} epochs:")
    header_line = " | ".join(h.rjust(w) for h, w in zip(headers, widths))
    separator = "-" * len(header_line)
    lines.extend([header_line, separator])
    for row in rows:
        line = " | ".join(cell.rjust(w) for cell, w in zip(row, widths))
        lines.append(line)
    lines.append(separator)

    epoch = ckpt.get('epoch', 'N/A')
    best = ckpt.get('best_val_loss', 'N/A')
    lines.append(f"Best validation loss: {best:.4f}" if isinstance(best, (int, float)) else f"Best validation loss: {best}")
    if isinstance(epoch, int) and 1 <= epoch <= len(accuracies):
        lines.append(f"Character recognition rate: {accuracies[epoch]:.2f}%")
    if cumulative:
        lines.append(f"Total training time: {cumulative[-1] / 60:.2f} minutes")
    opt = ckpt.get('optimizer_state_dict', {})
    if 'param_groups' in opt:
        lr = opt['param_groups'][0].get('lr', 'N/A')
        lines.append(f"Last used learning rate: {lr}")
    lines.append(separator)
    return "\n".join(lines)


if __name__ == '__main__':
    model_path = select_model()
    print(f"Selected model: {model_path}\n")
    stats = get_model_stats(model_path)
    print(stats)
