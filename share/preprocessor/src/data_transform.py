class DataTransform:
    @staticmethod
    def deep_get(_dict, keys, default=None):
        """
        Recursive function for finding a nested dict.
        _dict: the dictionary to search over
        keys: list of keys defining the nested path
        default: optional value to return when the given path is not found
        returns the nested dictionary
        """
        for key in keys:
            if isinstance(_dict, dict):
                _dict = _dict.get(key, default)
            else:
                return default
        return _dict

    @staticmethod
    def deep_put(_dict, keys, value):
        """
        A function for putting a given value at a key path.
        _dict: the input dictionary to modify
        keys: list of keys defining the nested path
        value: the value to add at the nested path
        returns the modified input dictionary
        NOTE: this takes advantage of the face that Python stores values by
        reference. Since `traverse_dict` is referencing the same memory as the
        input `_dict`, modifying it also modifies `_dict`.
        """
        traverse_dict = _dict
        for i, key in enumerate(keys[:-1]):
            traverse_dict = traverse_dict.get(key)
            if i == len(keys) - 2:
                traverse_dict[keys[-1]] = value
                break
        return _dict

    @staticmethod
    def to_list(arg):
        if type(arg) is not list:
            return [arg]
        return arg
