from BaseObject import BaseObject
from WakeCombination import WakeCombination
from WakeDeflection import WakeDeflection
from WakeVelocity import WakeVelocity


class Wake(BaseObject):

    def __init__(self):
        super().__init__()
        self.combination = WakeCombination("fls")
        self.deflection = WakeDeflection("jimenez")
        self.velocity = WakeVelocity("jensen")

    def solve(self):
        print("Wake")
    
    def valid(self):
        return True
    
    def getCombination(self):
        return self.combination
    
    def getDeflection(self):
        return self.deflection
    
    def getVelocity(self):
        return self.velocity

