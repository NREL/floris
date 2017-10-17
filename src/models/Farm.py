from BaseObject import BaseObject


class Farm(BaseObject):
    
    def __init__(self):
        super().__init__()
        
        self.turbineMap = None
        
    def valid(self):
        return True