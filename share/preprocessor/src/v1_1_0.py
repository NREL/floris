
from .version_class import VersionClass
from .data_transform import DataTransform


class V1_1_0(VersionClass, DataTransform):

    version_string = "v1.1.0"

    def __init__(self, meta_dict, turbine_dict, wake_dict, farm_dict):
        self.base_meta = meta_dict
        self.base_turbine = turbine_dict
        self.base_wake = wake_dict
        self.base_farm = farm_dict
        self._meta_dict = self.build_meta_dict()
        self._turbine_dict = self.build_turbine_dict()
        self._wake_dict = self.build_wake_dict()
        self._farm_dict = self.build_farm_dict()

    def build_meta_dict(self):
        self.base_meta["version"] = V1_1_0.version_string
        return self.base_meta

    def build_farm_dict(self):
        return self.base_farm

    def build_wake_dict(self):
        return self.base_wake

    def build_turbine_dict(self):
        return self.base_turbine

    @property
    def meta_dict(self):
        return self._meta_dict

    @property
    def turbine_dict(self):
        return self._turbine_dict

    @property
    def wake_dict(self):
        return self._wake_dict

    @property
    def farm_dict(self):
        return self._farm_dict
