from .data_synthesizer_dictionary import DataSynthesizerDictionary
from .data_synthesizer_cartesian import DataSynthesizerCartesian
from .data_synthesizer_random import DataSynthesizerRandom

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE_LETTERS = LETTERS.lower()
NUMBERS = "0123456789"
SPECIAL = " "

# EDIT THE SETTINGS OF IMAGE SYNTHESIZER HERE
params = {
    # MAIN CONFIGURATION =============================
    "NAME": "arial_times_general_20-28pt_8px",  # Name of the dataset
    "TYPE": "random",  # Which algorithm to use
    "N_IMAGES": 1000000,  # How many images to generate
    "MIN_LEN": 1,   # Shortest permitted string
    "MAX_LEN": 20,  # Longest permitted string
    "X": 512,  # Original image width
    "Y": 64,  # Original image height
    "MIN_FONT": 20,  # Smallest font used
    "MAX_FONT": 28,  # Largest font used
    "FONTS": ["C:/Windows/Fonts/ARIAL.ttf", "C:/Windows/Fonts/TIMES.ttf"],  # Fonts used
    # PIXELATION DETAILS =============================
    "MIN_PAD": 0,  # Smallest padding in px (applied top left)
    "MAX_PAD": 20,  # Largest padding in px (applied top left)
    "MIN_BLOCK": 8,  # Smallest block size of pixelation
    "MAX_BLOCK": 8,  # Largest block size of pixelation
    "SHIFT": True,  # Shift block grid randomly
    "ANTIALIAS": True,  # Antialias text or use binary rendering
    # RANDOM/CARTESIAN TYPE CONFIG =============================
    "CHARSET": LETTERS + LOWERCASE_LETTERS + NUMBERS + SPECIAL,  # Characters used for strings
    # CARTESIAN TYPE CONFIG =============================
    "SEQUENCE_LENGTH": 3,  # How long sequences to generate (in characters)
    # DICTIONARY TYPE CONFIG =============================
    "DICT_FILE": "english-words-466k.txt",  # Name of the dictionary file
}


def main():
    print(f"Generating dataset: {params['NAME']}")
    match params["TYPE"]:
        case "random":
            DataSynthesizerRandom(params)
        case "cartesian":
            DataSynthesizerCartesian(params)
        case "dictionary":
            DataSynthesizerDictionary(params)
        case _:
            raise Exception("Invalid synthesizer type!")


if __name__ == "__main__":
    main()
