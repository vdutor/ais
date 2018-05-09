import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm

from data.data import get_mnist_cvae, get_mnist_full_test
from deconvgp.cvae import CVAE, CVAE2, CVAE3

from deconvgp.vae import VAE, TF_VAE

# def Generator(CVAE3):

#     def __init__(self, *args, **kwargs):
#         super().__init__(self, *args, **kwargs)
#         self.label_dim = self.Y_dim  # mnist: 10
#         self.output_dim = self.X_dim  # mnist: 784 = 28^2
#         self.input_dim =  self.latent_dim + self.latent_dim

#     def __call__(self, ZY):
#         Z, Y = tf.split(ZY, [self.latent_dim, self.label_dim], axis=1)
#         return self._eval_decoder(Z, Y, training=False)

class Generator(object):

    def __init__(self, model):
        self.model = model
        self.output_dim = model.X_dim  # mnist: 784 = 28^2
        self.input_dim = model.latent_dim

    def __call__(self, Z):
        return self.model._build_decoder(Z)

class CGenerator(object):

    def __init__(self, model):
        self.model = model
        self.output_dim = model.X_dim  # mnist: 784 = 28^2
        self.latent_dim = model.latent_dim
        self.label_dim = model.Y_dim

    def __call__(self, Z, Y):
        return self.model._eval_decoder(Z, Y)


LATENT_DIM = 2
IMAGE_SIZE = [28, 28]
Nc = 4

X, Y = get_mnist_cvae(Nc)
Xs, Ys = get_mnist_full_test()
N_test = 10
# Xs = np.random.rand(5, 28**2)

vae = CVAE3(X, Y, LATENT_DIM, batch_size=64, dropout_rate=0.2)
# vae = VAE(X, LATENT_DIM)
model = CGenerator(vae)
prior = NormalPrior()
kernel = ParsenDensityEstimator()
sess = vae.enquire_session()
model = ais.CModel(model, prior, kernel, 0.25, 10, session=sess)

print("Conditional AIS")
schedule = ais.get_schedule(10, rad=4)
print((model.ais(Xs[:N_test, :], Ys[:N_test, :], schedule)))

# from gpflow.training import AdamOptimizer

# print("test_param", vae.test_param.value)
# print("MC")
# print(np.mean(vae.compute_test_log_likelihood(Xs[:N_test, :])))

# print("AIS")
# schedule = ais.get_schedule(10, rad=4)
# print(np.mean(model.ais(Xs[:N_test, :], schedule)))

# AdamOptimizer(0.001).minimize(vae, maxiter=100)

# print("test_param", vae.test_param.value)
# print("MC")
# print(np.mean(vae.compute_test_log_likelihood(Xs[:N_test, :])))

# print("AIS")
# schedule = ais.get_schedule(10, rad=4)
# print(np.mean(model.ais(Xs[:N_test, :], schedule)))



# # plt.plot(x, p1)
# # plt.plot(x, p2)
# # plt.show()
