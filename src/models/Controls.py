from BaseObject import BaseObject


class Controls(BaseObject):
    """
        Describe Controls here
    """

    def __init__(self,
                 turbineAngle=None,
                 yawAngle=None,
                 tiltAngle=None,
                 bladePitch=None):
        super().__init__()
        self.turbineAngle = turbineAngle
        self.yawAngle = yawAngle
        self.tiltAngle = tiltAngle
        self.bladePitch = bladePitch

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid
