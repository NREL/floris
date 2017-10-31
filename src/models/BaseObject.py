
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
        valid = True

        selfvalues = self.__dict__.values()
        if None in selfvalues:
            valid = False
            print("in ", self)
            for key in self.__dict__.keys():
                if self.__dict__[key] == None:
                    print(key + " has value None")

        return valid
