import random
from .data_synthesizer import DataSynthesizer


class DataSynthesizerRandom(DataSynthesizer):
    def get_text(self, index):
        # Use index as seed to make datasets deterministic to remove random variance from results
        rnd = random.Random(index)
        length = rnd.randint(self.MIN_LEN, self.MAX_LEN)
        return "".join(rnd.choices(self.CHARSET, k=length))
