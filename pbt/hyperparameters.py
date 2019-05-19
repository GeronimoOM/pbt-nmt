import logging
from abc import ABC, abstractmethod

import numpy as np
from keras import backend as K
from keras.layers import Dropout, GRU

log = logging.getLogger(__name__)


class Hyperparameter(ABC):
    """Base class of all hyperparameters that can be modified while training a
    model.

    """

    @abstractmethod
    def perturb(self, factors=None):
        """Perturb the hyperparameter with a random chosen factor.

        Args:
            factors (List[double]): factors to choose from.

        """
        pass

    @abstractmethod
    def replace_with(self, hyperparameter):
        """Replace the configuration of this hyperparameter with the
        configuration of the given one.

        Args:
            hyperparameter (Hyperparameter): hyperparameter to copy.

        """

        pass

    @abstractmethod
    def get_config(self):
        """Return the configuration (value(s)) for this hyperparameter.

        Returns:
             dict: dictionary where the key is the name of the hyperparameter.

        """
        pass


class DropoutHP(Hyperparameter, Dropout):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        self.rate = self.rate * np.random.choice(factors)

    def replace_with(self, hyperparameter):
        self.rate = hyperparameter.get_config().get('dr')

    def get_config(self):
        return {'dr': self.rate}


class GRUHP(Hyperparameter, GRU):

    def __init__(self, hidden_size, recurrent_dropout, **kwargs):
        super().__init__(hidden_size=hidden_size, recurrent_dropout=recurrent_dropout, **kwargs)

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        self.cell.recurrent_dropout = self.recurrent_dropout * np.random.choice(factors)
        self.cell._recurrent_dropout_mask = None

    def replace_with(self, hyperparameter):
        self.cell.recurrent_dropout = hyperparameter.get_config().get('rec_dr')
        self.cell._recurrent_dropout_mask = None

    def get_config(self):
        return {'rec_dr': self.cell.recurrent_dropout}


class FloatHyperparameter(Hyperparameter):

    def __init__(self, name, variable):
        self.name = name
        self.variable = variable

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        K.set_value(self.variable,
                    K.get_value(self.variable) * np.random.choice(factors))

    def replace_with(self, hyperparameter):
        K.set_value(self.variable, K.cast_to_floatx(
            hyperparameter.get_config().get(self.name)))

    def get_config(self):
        return {self.name: float(K.get_value(self.variable))}


def find_hyperparameters_model(keras_model):
    """Finds instances of class Hyperparameter that are used in the given model.

    For example, in the following model::

        keras.layers.Dense(
            42,
            kernel_regularizer=pbt.hyperparameters.L1L2Mutable(l1=0, l2=1e-5),
            bias_initializer=keras.initializers.Zeros(),
            input_shape=(13,)
        )

    L1L2Mutable is an instance of pbt.hyperparameters.Hyperparameter, but
    l1_l2 is not. As a result, the method will only return the former.

    Args:
        keras_model (keras.models.Sequential): a compiled Keras model.

    Returns:
        A list of hyperparameters.

    """
    hyperparameters = []
    for layer in keras_model.layers:
        if isinstance(layer, Hyperparameter):
            hyperparameters.append(layer)
        else:
            hyperparameters.extend(find_hyperparameters_layer(layer))
    return hyperparameters


def find_hyperparameters_layer(keras_layer):
    """Finds instances of class Hyperparameter that are used in the given layer.

    For example, in the following model::

        keras.layers.Dense(
            42,
            kernel_regularizer=pbt.hyperparameters.L1L2Mutable(l1=0, l2=1e-5),
            bias_initializer=keras.initializers.Zeros(),
            input_shape=(13,)
        )

    L1L2Mutable is an instance of pbt.hyperparameters.Hyperparameter, but
    Zeros is not. As a result, the method will only return the former.

    Args:
        keras_layer (keras.layers.Layer): a Keras layer object.

    Returns:
        A list of hyperparameters.

    """
    hyperparameters_names = ['kernel_regularizer']
    hyperparameters = []
    for h_name in hyperparameters_names:
        if hasattr(keras_layer, h_name):
            h = getattr(keras_layer, h_name)
            if isinstance(h, Hyperparameter):
                hyperparameters.append(h)
    return hyperparameters
