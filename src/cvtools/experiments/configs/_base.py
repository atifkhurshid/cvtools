"""
Base Class for Configurations
"""

# Author: Atif Khurshid
# Created: 2026-05-08
# Modified: 2026-05-15
# Version: 1.1
# Changelog:
#     - 2026-05-08: Initial version.
#     - 2026-05-15: Changed config model from singleton to regular class.
#     - 2026-05-15: Added _update method for updating config values from kwargs.

import json


class ConfigBase:
    """
    Base class for configurations.
    """
    def __init__(self):
        pass


    def __repr__(self):
        """
        Return a JSON-formatted string representation of the configuration.
        """
        repr = json.dumps(self.__dict__, indent = 4, default = str)
        repr = repr.replace("\\n", "\n")

        return repr


    def _update_from_kwargs(self, **kwargs):
        """
        Update config values from kwargs.

        Should be run at the end of the __init__ method of all config classes
        to allow for overriding default values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config key: '{key}'")
            