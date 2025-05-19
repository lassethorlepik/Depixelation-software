import datetime
import builtins
import glob
import os
import random
import string
from colorama import init, Fore
import torch
import numpy as np
import Levenshtein

init()  # Initializes colorama to make ANSI escape character sequences work under MS Windows as well


def print(text):
    "Print string as a fancy timestamped log."
    # Get current time with microseconds
    now = datetime.datetime.now()
    # Format time as hh:mm:ss:ms
    formatted_time = now.strftime("%H:%M:%S") + f':{now.microsecond // 1000:03d}'
    # Print the text with a colored timestamp
    builtins.print(f"{Fore.YELLOW}[{formatted_time}]{Fore.RESET} {text}")


INVALID_CHARS = '\\/*?:"<>|'
_TRANSLATION_TABLE = str.maketrans('', '', INVALID_CHARS)


def sanitize_filename(string):
    return string.translate(_TRANSLATION_TABLE).strip()


def strip_chars(string, charset):
    allowed = set(charset)
    return ''.join(c for c in string if c in allowed)


def check_gpu_compute():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
    return device


def remove_prefix(text):
    """Remove the prefix from training input images."""
    return text.split("_")[1]


def generate_random_string(length=4, letters=string.ascii_letters + string.digits):
    return ''.join(random.choice(letters) for _ in range(length))


def delete_png_files(folder_path):
    pattern = os.path.join(folder_path, '*.png')
    for file_path in glob.glob(pattern):
        os.remove(file_path)


def add_noise(img, noise_factor=1):
    if torch.rand(1).item() < 0.5:
        return img
    noise = noise_factor * torch.rand(img.size()) * 0.002
    noise = torch.randn(img.size()) * noise
    img = img + noise
    img = torch.clamp(img, 0.0, 1.0)
    return img


def apply_compression(img, quality=70):
    if torch.rand(1).item() < 0.5:
        return img
    # Simulate JPEG compression using PIL
    from PIL import Image
    import io
    img_np = img.squeeze(0).numpy() * 255
    img_pil = Image.fromarray(img_np.astype(np.uint8), mode='L')
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer).convert('L')
    img_compressed = np.array(img_compressed).astype(np.float32) / 255.0
    img_compressed = torch.from_numpy(img_compressed).unsqueeze(0)
    return img_compressed


def random_shift(img, max_shift=3):
    pad_top = torch.randint(0, max_shift + 1, (1,)).item()
    pad_left = torch.randint(0, max_shift + 1, (1,)).item()

    # Pad the image with white (1) on the top and left
    pad = (pad_left, 0, pad_top, 0)  # (left, right, top, bottom)
    img_padded = torch.nn.functional.pad(img, pad, mode='constant', value=1.0)

    # Crop to original size
    _, H, W = img.size()
    img_shifted = img_padded[:, :H, :W]
    return img_shifted


def downscale_image(img, img_width, img_height):
    # Downscale using interpolation
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_height, img_width), mode='nearest').squeeze(0)
    return img


def crop_white(image):
    """Crop the image tensor by removing irrelevant white background."""
    mask = ~((image == 1).all(dim=0))  # shape: [H, W]
    # If the entire image is white, return the original image.
    if not mask.any():
        return image
    # Find the bounding box of the non-white content.
    nonzero = torch.nonzero(mask, as_tuple=False)
    y_min = nonzero[:, 0].min().item()
    y_max = nonzero[:, 0].max().item() + 1  # +1 for slicing
    x_min = nonzero[:, 1].min().item()
    x_max = nonzero[:, 1].max().item() + 1
    return image[:, y_min:y_max, x_min:x_max]


def pad_custom_color(img, target_width, target_height, pad_color=1.0):
    # Get current dimensions
    _, H, W = img.size()
    # If image is larger than target, crop it (omit overflow)
    if H > target_height:
        img = img[:, :target_height, :]
    if W > target_width:
        img = img[:, :, :target_width]
    # Update dimensions after cropping
    _, H, W = img.size()
    # Compute needed padding for each side
    pad_height = target_height - H
    pad_width = target_width - W
    # Ensure padding is non-negative
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)
    # pad is defined as (left, right, top, bottom)
    pad = (0, pad_width, 0, pad_height)
    img_padded = torch.nn.functional.pad(img, pad, mode='constant', value=pad_color)
    return img_padded


def similarity(str1, str2):
    """Evaluate string similarity."""
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 100  # Both strings are empty
    similarity_score = (1 - distance / max_len) * 100
    return similarity_score


def decode_label(num_to_char, label_tensor):
    chars = [num_to_char.get(num.item(), '') for num in label_tensor if num != 0]
    return ''.join(chars)
