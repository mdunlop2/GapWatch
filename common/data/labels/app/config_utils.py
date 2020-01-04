import json
from collections import OrderedDict
import os
from stat import * # ST_SIZE etc
from datetime import datetime
from copy import deepcopy


class JSONPropertiesFileError(Exception):
    pass


class JSONPropertiesFile(object):

    def __init__(self, file_path, default={}):
        self.file_path = file_path
        self._default_properties = default
        self._validate_file_path(file_path)

    def _validate_file_path(self, file_path):
        if not file_path.endswith(".json"):
            raise JSONPropertiesFileError(f"Must be a JSON file: {file_path}")
        if not os.path.exists(file_path):
            self.set(self._default_properties)

    def set(self, properties):
        new_properties = deepcopy(self._default_properties)
        new_properties.update(properties)
        with open(self.file_path, 'w') as file:
            json.dump(new_properties, file, indent=4)


    def get(self):
        properties = deepcopy(self._default_properties)
        with open(self.file_path) as file:
            properties.update(json.load(file, object_pairs_hook=OrderedDict))
        return properties

    def get_file_info(self):
        st = os.stat(self.file_path)
        res = {
            'size':st[ST_SIZE],
            'size_str':str(round(st[ST_SIZE]/1000,2)) + ' KB',
            'last_mod': datetime.fromtimestamp(st[ST_MTIME]).strftime("%Y-%m-%d")
        }
        return res

