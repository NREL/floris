from BaseObject import BaseObject


class Controls(BaseObject):
    """
        Describe Controls here
    """

    def __init__(self,
                 turbine_angle=None,
                 yaw_angle=None,
                 tilt_angle=None,
                 blade_pitch=None):
        super().__init__()
        self.turbineAngle = turbine_angle
        self.yawAngle = yaw_angle
        self.tiltAngle = tilt_angle
        self.bladePitch = blade_pitch

    def valid(self):
        """
            Do validity check
        """
        valid = True
        if not super().valid():
            valid = False
        return valid
