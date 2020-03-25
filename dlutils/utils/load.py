from collections import Mapping, Sequence
from pydoc import locate

from omegaconf import Config, OmegaConf

__all__ = ['load_objects']


def load_objects(to_load):
    if isinstance(to_load, Config):
        to_load = OmegaConf.to_container(to_load)

    if isinstance(to_load, Mapping):
        # using 'cls' as an indicator to have a class, that should be
        # initialized with parameters. If the object should only be loaded
        # and not called/initialized, it has to be passed as raw string
        # without the 'cls'-key in a subdict
        if 'cls' in to_load and isinstance(to_load['cls'], str):
            dict_copy = {k: v for k, v in to_load.items() if k != 'cls'}
            try:
                # load the class object
                class_obj = locate(to_load['cls'])

                # There are two ways to specify parameters:
                # 1.) as a subdict 'params', which is handled inside this
                #   if-statement
                # 2.) Using all the other kwargs on the same dict level as
                #   parameters (which is handled inside the else-statement).
                # Per default, when there is a subdict 'params' it will be
                # used and the other dict entries on the same level will be
                # ignored
                if 'params' in dict_copy and isinstance(dict_copy['params'],
                                                        Mapping):
                    in_params = True
                    params = load_objects(
                        dict_copy['params'])

                else:
                    in_params = False
                    params = load_objects(dict_copy)

                # Module could not be loaded (ImportError is handled by pydoc)
                if class_obj is None:
                    loaded_obj = {'cls': load_objects(to_load['cls'])}

                    # Switch output way
                    if in_params:
                        loaded_obj.update(params=params)
                    else:
                        loaded_obj.update(params)

                    return loaded_obj

                # initialize class with loaded parameters
                return class_obj(**params)
            except (ModuleNotFoundError, ImportError):
                pass

        # recursively apply it to all items in the mapping
        return type(to_load)({k: load_objects(v)
                              for k, v in to_load.items()})
    # namedtuple
    elif isinstance(to_load, tuple) and hasattr(to_load, '_fields'):
        return type(to_load)(*(load_objects(x) for x in to_load))
    # for strings: load class or function if possible
    elif isinstance(to_load, str):
        try:
            module = locate(to_load)
            # Module could not be loaded (ImportError is handled by pydoc)
            if module is None:
                return to_load
        except (ModuleNotFoundError, ImportError):
            return to_load

    # recursively apply it to all items in sequence
    elif isinstance(to_load, Sequence):
        return type(to_load)([load_objects(_item)
                              for _item in to_load])

    return to_load
