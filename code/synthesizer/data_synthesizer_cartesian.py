from .data_synthesizer import DataSynthesizer


class DataSynthesizerCartesian(DataSynthesizer):
    def __init__(self, params):
        super().__init__(params)
        self.charset = params["CHARSET"]
        self.n = params["SEQUENCE_LENGTH"]
        self.k = len(self.charset)
        self.total_combinations = self.k ** self.n

    def get_text(self, index):
        """Fetch combination of characters based on index."""
        if index < 0 or index >= self.total_combinations:
            return None

        # Convert index to base-k with exactly self.n digits
        s = []
        rem = index
        for pos in range(self.n):
            exp = self.n - pos - 1
            base = self.k ** exp
            digit = rem // base
            s.append(self.charset[digit])
            rem %= base

        return "".join(s)
