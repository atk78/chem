import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from itertools import cycle
from adjustText import adjust_text

from . import utils
from .models import LSTMAttention
from .features import token, augm


def visualize_embedding(
    data_name,
    smiles_toviz="CCC",
    augmentation=False,
    outdir="reports",
    affinity_propn=True,
):
    pass
