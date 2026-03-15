from .gbt import GBT
from .gp import GP
from .knn import KNN
from .nn import NN
from .rf import RF
from .sv import SV

def get_model(model_name):

    if model_name == 'gbt':
        model = GBT()
    if model_name == 'gp':
        model = GP()
    if model_name == 'gp_ard':
        model = GP(isotropic=False)
    if model_name == 'knn':
        model = KNN()
    if model_name == 'nn':
        model = NN()
    if model_name == 'rf':
        model = RF()
    if model_name == 'sv':
        model = SV()

    return model