import json
import os
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont
import random
import util
from util import print


class DataSynthesizer:
    def __init__(self, params):
        print("Starting synthesizer...")
        for key, value in params.items():
            setattr(self, key, value)
        self.FINAL_X = self.X // self.MIN_BLOCK
        self.FINAL_Y = self.Y // self.MIN_BLOCK
        self.folder = f"../datasets/{params['NAME']}"
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(f"{self.folder}/debug", exist_ok=True)
        os.makedirs(f"{self.folder}/images", exist_ok=True)
        util.delete_png_files(f"{self.folder}/debug")
        util.delete_png_files(f"{self.folder}/images")
        self.create_metadata(params)
        print("Precomputing...")
        self.precompute()
        print("Creating debug images...")
        self.create_debug_images()
        print("Starting multiprocessing...")
        self.multiprocess()

    def precompute(self):
        "Ran just before multiprocessing is started. Some synthesizers precompute values here"
        pass

    def multiprocess(self):
        "Distribute tasks to all cpu cores."
        tasks = []
        num_processes = cpu_count()
        images_folder = f"{self.folder}/images"
        images_per_process = self.N_IMAGES // num_processes
        ranges = [(i * images_per_process, (i + 1) * images_per_process) for i in range(num_processes)]
        # Adjust last chunk for any remainder
        ranges[-1] = (ranges[-1][0], self.N_IMAGES)
        for i, (start_idx, end_idx) in enumerate(ranges):
            tasks.append((start_idx, end_idx, images_folder, (i == 0)))
        with Pool(processes=num_processes) as pool:
            pool.starmap(self.create_images_for_range, tasks)

    def create_metadata(self, params):
        "Generate a file containing dataset generation parameters"
        with open(f"{self.folder}/metadata.json", "w") as file:
            json.dump(params, file, indent=4)

    def create_image(self, index, output_folder, text):
        """Process a single string into a pixelated image using index as seed."""
        random.seed(index)

        block_size = random.randint(self.MIN_BLOCK, self.MAX_BLOCK)
        font_size = random.randint(self.MIN_FONT, self.MAX_FONT)
        font = ImageFont.truetype(random.choice(self.FONTS), font_size)

        background_color = 255
        text_color = 0
        bitmap = Image.new('L', (self.X, self.Y), background_color)
        draw = ImageDraw.Draw(bitmap)
        draw.fontmode = self.get_antialias_type(index)

        pad_x = random.randint(self.MIN_PAD, self.MAX_PAD)
        pad_y = random.randint(self.MIN_PAD, self.MAX_PAD)
        draw.text((pad_x, pad_y), text, font=font, fill=text_color)

        if self.SHIFT:
            shiftx = random.randint(-self.MAX_BLOCK, self.MAX_BLOCK)
            shifty = random.randint(-self.MAX_BLOCK, self.MAX_BLOCK)
        else:
            shiftx = shifty = 0

        self.pixelate_image_mono(bitmap, block_size, shift_x=shiftx, shift_y=shifty)
        bitmap = self.crop_and_pad_image(bitmap, block_size, background_color)

        if all(p == background_color for p in bitmap.getdata()):
            print(f"Skipping index {index!r}: no non-white pixels generated.")
            return

        file_name = f"{output_folder}/{index}_{text}.png"
        bitmap.save(file_name)

    def get_antialias_type(self, index):
        random.seed(index)
        if self.ANTIALIAS == True:
            return "L"
        elif self.ANTIALIAS == False:
            return "1"
        else:
            return "L" if random.choice([True, False]) else "1"

    def get_text(self, index):
        "Fetch text for a specific index in the dataset"
        return "STRING"

    def create_images_for_range(self, start_index, end_index, output_folder, report_progress=False):
        "Generate images in given task"
        total_images = end_index - start_index
        progress_report_interval = max(1, total_images // 100)
        for i in range(start_index, end_index):
            text = self.get_text(i)
            if text is None:
                return  # Allows text generator to manually finish generation process
            text = util.sanitize_filename(text)
            self.create_image(i, output_folder, text)
            if report_progress and (i - start_index) % progress_report_interval == 0:
                progress_percent = ((i - start_index) / total_images) * 100
                print(f"Progress: {progress_percent:.1f}%")

    def create_debug_images(self, outline=False):
        """Create special images for debugging."""
        output_folder = f"{self.folder}/debug"
        text = util.sanitize_filename(self.get_text(0))
        background_color = 255
        text_color = 0
        block_size = self.MAX_BLOCK
        font_size = self.MAX_FONT
        random.seed(0)
        font = ImageFont.truetype(random.choice(self.FONTS), font_size)
        bitmap = Image.new('L', (self.X, self.Y), background_color)
        draw = ImageDraw.Draw(bitmap)
        draw.fontmode = self.get_antialias_type(0)
        draw.text((random.randint(self.MIN_PAD, self.MAX_PAD), random.randint(self.MIN_PAD, self.MAX_PAD)), text,
                  font=font, fill=text_color)
        file_name = f"{output_folder}/txt_max_{text}.png"
        bitmap.save(file_name)
        self.pixelate_image_mono(bitmap, block_size, shift_x=self.MAX_PAD, shift_y=self.MAX_PAD)
        if outline:
            bbox = self.find_bounding_box(bitmap)
            draw.rectangle(bbox, outline='red', width=3)
        file_name = f"{output_folder}/pix_max_{text}.png"
        bitmap.save(file_name)
        bitmap = self.crop_and_pad_image(bitmap, block_size, background_color)
        file_name = f"{output_folder}/crp_max_{text}.png"
        bitmap.save(file_name)
        # Second version
        text = util.sanitize_filename(self.get_text(1))
        block_size = self.MIN_BLOCK
        font_size = self.MIN_FONT
        random.seed(1)
        font = ImageFont.truetype(random.choice(self.FONTS), font_size)
        bitmap = Image.new('L', (self.X, self.Y), background_color)
        draw = ImageDraw.Draw(bitmap)
        draw.fontmode = self.get_antialias_type(1)
        draw.text((random.randint(self.MIN_PAD, self.MAX_PAD), random.randint(self.MIN_PAD, self.MAX_PAD)), text,
                  font=font, fill=text_color)
        file_name = f"{output_folder}/txt_min_{text}.png"
        bitmap.save(file_name)
        self.pixelate_image_mono(bitmap, block_size, shift_x=self.MIN_PAD, shift_y=self.MIN_PAD)
        if outline:
            bbox = self.find_bounding_box(bitmap)
            draw.rectangle(bbox, outline='red', width=1)
        file_name = f"{output_folder}/pix_min_{text}.png"
        bitmap.save(file_name)
        bitmap = self.crop_and_pad_image(bitmap, block_size, background_color)
        file_name = f"{output_folder}/crp_min_{text}.png"
        bitmap.save(file_name)

    def pixelate_image_mono(self, image, block_size, shift_x=0, shift_y=0):
        width, height = image.size
        pixels = image.load()
        # Adjust shifts to be within block boundaries.
        shift_x %= block_size
        shift_y %= block_size
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                intensity_sum = 0
                count = 0
                x_start = x + shift_x
                y_start = y + shift_y
                for iy in range(block_size):
                    for ix in range(block_size):
                        pos_x = x_start + ix
                        pos_y = y_start + iy
                        if 0 <= pos_x < width and 0 <= pos_y < height:
                            intensity = pixels[pos_x, pos_y]
                        else:
                            intensity = 255
                        intensity_sum += intensity
                        count += 1
                avg_intensity = intensity_sum // count if count else 255
                for iy in range(block_size):
                    for ix in range(block_size):
                        pos_x = x + ix
                        pos_y = y + iy
                        if 0 <= pos_x < width and 0 <= pos_y < height:
                            pixels[pos_x, pos_y] = avg_intensity

    def find_bounding_box(self, image):
        background_color = 255
        width, height = image.size
        left, top, right, bottom = width, height, -1, -1
        pixels = image.load()
        for y in range(height):
            for x in range(width):
                if pixels[x, y] != background_color:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
        if right == -1 or bottom == -1:
            return 0, 0, width, height
        return left, top, right, bottom

    def crop_and_pad_image(self, image, block_size, background_color):
        bbox = self.find_bounding_box(image)
        cropped_image = image.crop(bbox)
        new_width = (cropped_image.width + block_size - 1) // block_size
        new_height = (cropped_image.height + block_size - 1) // block_size
        scaled_image = cropped_image.resize((new_width, new_height), Image.NEAREST)
        padded_image = Image.new('L', (self.FINAL_X, self.FINAL_Y), background_color)
        padded_image.paste(scaled_image, (0, 0))
        return padded_image
