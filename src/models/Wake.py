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
    
