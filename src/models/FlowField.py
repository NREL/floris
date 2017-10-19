from BaseObject import BaseObject


class FlowField(BaseObject):
    """
        Describe FF here
    """

    def __init__(self, wakeCombination=None):
        super().__init__()
        self.wakeCombination = wakeCombination

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid
