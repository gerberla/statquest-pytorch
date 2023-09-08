from torch.types import Number

class LinSpace:
    start: Number
    end: Number
    step: int = None

    def __init__(self, start: Number, end: Number, step: int=1):
        self.start=start
        self.end=end
        self.step=step