from BaseObject import BaseObject


class Farm(BaseObject):
    """
        Describe farm here
    """

    def __init__(self):
        super().__init__()
        self.turbineMap = None
        self.wake = None
        self.wakeCombination = None

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid
