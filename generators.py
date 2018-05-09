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

