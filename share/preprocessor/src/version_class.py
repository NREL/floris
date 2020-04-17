
from .output import Output
from abc import ABC, abstractmethod

class VersionClass(ABC):
    version_string = "v0.0.0"

    @abstractmethod
    def build_meta_dict(self):
        pass

    @abstractmethod
    def build_farm_dict(self):
        pass

    @abstractmethod
    def build_wake_dict(self):
        pass

    @abstractmethod
    def build_turbine_dict(self):
        pass

    @abstractmethod
    def meta_dict(self):
        pass

    @abstractmethod
    def farm_dict(self):
        pass

    @abstractmethod
    def turbine_dict(self):
        pass

    @abstractmethod
    def wake_dict(self):
        pass
    
    def build_input_file_data(self):
        return {
            **self.meta_dict,
            **self.farm_dict,
            **self.turbine_dict,
            **self.wake_dict
        }

    def export(self, filename=version_string + ".json"):
        output = Output(filename)
        file_data = self.build_input_file_data()
        output.write_dictionary(file_data)
        output.end()

    def input_or_default(self, key, search_dictionary, default_dictionary):
        if key in search_dictionary:
            return search_dictionary[key]
        else:
            return default_dictionary[key]
