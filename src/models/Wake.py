from BaseObject import BaseObject
from WakeDeflection import WakeDeflection
from WakeVelocity import WakeVelocity


class Wake(BaseObject):

    def __init__(self):
        super().__init__()
        self.deflection = None
        self.velocity = None

    def valid(self):
        """
            Implement property checking here
        """
        valid = True
        if not super().valid():
            valid = False
        return valid

    def getDeflection(self):
        return self.deflection

    def getVelocity(self):
        return self.velocity

    def V(self, u, x, y, z):
        # yDisp, zDisp = self.displ(x)
        # wSlice = self.wakeSlicer(U, x, Y-yDisp, Z-zDisp)
        # 
        # # Broadcast x to make a matrix: (x + 0*y) = np.broadcast_to(x, y.shape)
        # tempCoords = np.einsum('ij, jmn -> imn',
        #                        self.Ri, np.stack([x + 0*Y, Y, Z]))
        # rotorBoundary = self.tanHFilter(tempCoords[0], 0, 1)
        # wakeBoundary = self.tanHFilter(wSlice.BT, wSlice.BV, 1)
        # mask = rotorBoundary*wakeBoundary
        # 
        # return wSlice.V*mask + U*(1-mask)
        return 1.0