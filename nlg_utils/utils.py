import os
import pickle

import numpy as np


class DirectoryHelper:

    def __init__(self, arg):
        if arg is None or isinstance(arg, str):
            self.path = arg
        elif isinstance(arg, DirectoryHelper):
            self.path = arg.path
        else:
            raise ValueError

    def get_path(self, filename: str):
        return os.path.join(self.path, filename) if self.path else None

    def get_directory(self, dirname: str):
        return self.__class__(self.get_path(dirname))

    def makedirs(self):
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)


class FileCache:

    @classmethod
    def tofile(cls, path, makedirs: bool = True):
        def decorator(func):
            def new_func(*args, **kwargs):
                path_str = path(*args, **kwargs) if callable(path) else path
                if not path_str:
                    return func(*args, **kwargs)

                if os.path.isfile(path_str):
                    print(f"load from '{path_str}'")
                    return cls.load_data(path_str)

                data = func(*args, **kwargs)
                if makedirs:
                    os.makedirs(os.path.dirname(path_str), exist_ok=True)
                print(f"cache to '{path_str}'")
                cls.save_data(data, path_str)
                return data
            return new_func

        return decorator

    def load_data(path):
        raise NotImplementedError

    def save_data(data, path):
        raise NotImplementedError


class PickleCache(FileCache):

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as f_in:
            return pickle.load(f_in)

    @staticmethod
    def save_data(data, path):
        with open(path, 'wb') as f_out:
            pickle.dump(data, f_out)


class NumpyCache(FileCache):

    @staticmethod
    def load_data(path):
        return np.load(path)['data']

    @staticmethod
    def save_data(data, path):
        np.savez_compressed(path, data=data)
