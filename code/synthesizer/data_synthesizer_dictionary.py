from util import strip_chars
from .data_synthesizer import DataSynthesizer


class DataSynthesizerDictionary(DataSynthesizer):
    def __init__(self, params):
        super().__init__(params)
        self.string_dict = {}

    def get_text(self, index):
        if index < 0 or index >= len(self.string_dict):
            return None
        return self.string_dict[index].strip()

    def precompute(self):
        with open("../strings/" + self.DICT_FILE, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file):
                self.string_dict[index] = strip_chars(line.strip(), self.CHARSET)
