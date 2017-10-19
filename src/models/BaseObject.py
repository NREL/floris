
class BaseObject(object):
    """
        The BaseObject class is the basis for all other classes.

        It is mainly a placeholder for future features. Currently implemented
        in the base class is

        valid(self) -> bool
            checks object validity based on null attributes
    """
    
    def __init__(self):
        self.placeholder = "this is a placeholder for future use"

    def valid(self):
        return None not in self.__dict__.values()
