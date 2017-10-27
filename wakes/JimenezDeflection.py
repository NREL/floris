import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..', 'src', 'models')))
import src.models.Wake as Wake


class JimenezDeflection(Wake):
    """
        Describe Jensen deflection model here
    """

    def __init__(self):
        super(JimenezDeflection, self).__init__()


class jimenezDeflection:
    """This class instantiates an object for computing the downwind
    deflection of a wake according to Jimenez et al"""

    def __init__(self, model, layout, cSet, output, turbI):
        # Extract the model properties from model and set them in the class
        self.kd = model.wakeDeflection
        self.ad = model.ad
        self.bd = model.bd
        self.aT = model.aT
        self.bT = model.bT

        self.D = layout.turbines[turbI].rotorDiameter
        self.Ct = output.Ct[turbI]
        self.yaw = np.radians(cSet.yawAngles[turbI])
        self.tilt = np.radians(cSet.tiltAngles[turbI])

        # angle of deflection
        self.xiInitYaw = 1./2.*np.cos(self.yaw)*np.sin(self.yaw)*self.Ct
        self.xiInitTilt = 1./2.*np.cos(self.tilt)*np.sin(self.tilt)*self.Ct
        # xi = the angle at location x, this expression is not further used,
        # yYaw uses the second order taylor expansion of xi.
        # xiYaw = (xiInitYaw)/(( 1 + 2*kd*(x/D) )**2)

    def displ(self, x):
        # yaw displacement
        displYaw = ((self.xiInitYaw * (15*((2*self.kd*x/self.D) + 1)**4 +
                    self.xiInitYaw**2)/((30*self.kd/self.D)*(2*self.kd*x /
                     self.D + 1)**5.)) - (self.xiInitYaw*self.D*(15 +
                                          self.xiInitYaw**2.)/(30*self.kd)))
        # corrected yaw displacement with lateral offset
        displYawTotal = displYaw + (self.ad + self.bd*x)

        displTilt = ((self.xiInitTilt * (15*((2*self.kd*x/self.D) + 1)**4 +
                     self.xiInitTilt**2)/((30*self.kd/self.D)*(2*self.kd*x /
                      self.D + 1)**5.)) - (self.xiInitTilt*self.D*(15 +
                                           self.xiInitTilt**2.)/(30*self.kd)))
        displTiltTotal = displTilt + (self.aT + self.bT*x)
        return displYawTotal, displTiltTotal
    