from importlib import import_module


class ModelManager(object):
    """Simple factory class to manage the models"""
    def create(network, num_layers):
        net = import_module('models.' + network)
        sym = net.get_symbol(num_layers)
        return sym
    create = staticmethod(create)
    
    
    