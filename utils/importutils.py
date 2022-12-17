"""Lazy imports using importlib.util.LazyLoader

Source: https://docs.python.org/3/library/importlib.html#importlib.util.LazyLoader
"""
import sys
import importlib.util


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    
    return module
