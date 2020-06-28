"""
Class reworked to work with PyTorch
"""

from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn

# ----------------
BNN_LAYER_TAG = 'BNNLayer'
USE_REPARAMETRIZATION_TRICK = True
PI_tensor = torch.Tensor([np.pi])
# ----------------

def square(x):
    return x**2

class BNNLayer(nn.Module):
    """Probabilistic layer that uses Gaussian weights.
    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=nn.ReLU(),
                 prior_sd=None,
                 **kwargs):
        super(BNNLayer, self).__init__()

        # self._srng = RandomStreams() TODO: check

        # Set vars.
        self.nonlinearity = nonlinearity
        self.num_inputs = incoming
        self.num_units = num_units
        self.prior_sd = prior_sd

        prior_rho = self.std_to_log(self.prior_sd)

        self.W = torch.Tensor(np.random.normal(0., prior_sd,
                                  (self.num_inputs, self.num_units)))  # @UndefinedVariable
        self.b = torch.Tensor(np.zeros(
            (self.num_units,),
            dtype=np.float))

        # Here we set the priors.
        # -----------------------
        mu = torch.Tensor(self.num_inputs, self.num_units)
        torch.nn.init.normal_(mu, mean=0., std=1.)
        self.mu = nn.Parameter(mu)

        rho = torch.Tensor(self.num_inputs, self.num_units)
        torch.nn.init.constant_(rho, prior_rho.item())
        self.rho = nn.Parameter(rho)

        # Bias priors.
        b_mu = torch.Tensor(self.num_units)
        torch.nn.init.normal_(b_mu, mean=0., std=1.)
        self.b_mu = nn.Parameter(b_mu)

        b_rho = torch.Tensor(self.num_units)
        torch.nn.init.constant_(b_rho, prior_rho.item())
        self.b_rho = nn.Parameter(b_rho)

        # -----------------------

        # Backup params for KL calculations.

        self.mu_old = torch.Tensor(np.zeros((self.num_inputs, self.num_units)))
        self.rho_old = torch.Tensor(np.ones((self.num_inputs, self.num_units)))
        self.b_mu_old = torch.Tensor(np.zeros((self.num_units,)))
        self.b_rho_old = torch.Tensor(np.ones((self.num_units,)))

    def log_to_std(self, rho):
        """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+
        This makes sure that we don't get negative stds. However, a downside might be
        that we have little gradient on close to 0 std (= -inf using this transformation).
        """
        return torch.log(1 + torch.exp(rho))

    def std_to_log(self, sigma):
        """Reverse log_to_std transformation."""
        return torch.log(torch.exp(sigma) - 1)

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = torch.Tensor(self.num_inputs, self.num_units)
        torch.nn.init.normal_(epsilon, mean=0., std=1.)
        # epsilon = torch.autograd.Variable(epsilon) # TODO: check if needed

        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + self.log_to_std(self.rho) * epsilon
        self.W = W
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = torch.Tensor(self.num_units, )
        torch.nn.init.normal_(epsilon, mean=0., std=1.)
        # epsilon = torch.autograd.Variable(epsilon) # TODO: check if needed

        b = self.b_mu + self.log_to_std(self.b_rho) * epsilon
        self.b = b
        return b

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.
        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.
        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.view(-1, self.num_inputs)

        gamma = torch.mm(input, self.mu) + self.b_mu.expand(input.size()[0], self.num_units)
        delta = torch.mm(square(input), square(self.log_to_std(
            self.rho))) + square(self.log_to_std(self.b_rho)).expand(input.size()[0], self.num_units)

        epsilon = torch.Tensor(self.num_units, )
        torch.nn.init.normal_(epsilon, mean=0., std=1.)
        # epsilon = torch.autograd.Variable(epsilon) # TODO: check if needed

        activation = gamma + torch.sqrt(delta) * epsilon

        return self.nonlinearity(activation)

    def forward(self, x, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(x, **kwargs)
        else:
            return self.get_output_for_default(x, **kwargs)

    def save_old_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old = self.mu.clone()
        self.rho_old = self.rho.clone()
        self.b_mu_old = self.b_mu.clone()
        self.b_rho_old = self.b_rho.clone()

    def reset_to_old_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu.data = self.mu_old.data
        self.rho.data = self.rho_old.data
        self.b_mu.data = self.b_mu_old.data
        self.b_rho.data = self.b_rho_old.data

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = square(p_mean - q_mean) + \
            square(p_std) - square(q_std)
        denominator = 2 * square(q_std) + 1e-8
        return torch.sum(
            numerator / denominator + torch.log(q_std) - torch.log(p_std))

    def kl_div_new_old(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))
        kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho),
                                  self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div_old_new(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), torch.tensor([0.]), self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu,
                                  self.log_to_std(self.b_rho), torch.tensor([0.]), self.prior_sd)
        return kl_div

    def kl_div_old_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), torch.tensor([0.]), self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), torch.tensor([0.]), self.prior_sd)
        return kl_div

    def kl_div_prior_new(self):
        kl_div = self.kl_div_p_q(
            torch.tensor([0.]), self.prior_sd, self.mu,  self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(torch.tensor([0.]), self.prior_sd,
                                  self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

    def get_output_for(self, input, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(input, **kwargs)
        else:
            return self.get_output_for_default(input, **kwargs)

    def get_output_for_default(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.view(-1, self.num_inputs)

        activation = torch.mm(input, self.get_W()) + \
            self.get_b().expand(input.size()[0], self.num_units)

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class BNN(nn.Module):
    """Bayesian neural network (BNN) based on Blundell2016."""

    def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 n_batches,
                 layers_type=None, #TODO
                 trans_func=nn.ReLU(),
                 out_func=nn.Identity(),
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 likelihood_sd=5.0,
                 second_order_update=False,
                 learning_rate=0.0001,
                 compression=False,
                 information_gain=True,
                 ):

        super(BNN, self).__init__()

        # assert len(layers_type) == len(n_hidden) + 1 TODO

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = torch.Tensor([prior_sd])
        self.layers_type = layers_type
        self.n_batches = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.likelihood_sd = torch.Tensor([likelihood_sd])
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.compression = compression
        self.information_gain = information_gain

        assert self.information_gain or self.compression

        # Build network architecture.
        self.build_network()
        self.opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def build_network(self):
        # TODO: extend for non-Bayesian layers
        self.layers = torch.nn.ModuleList()
        # Build network architecture
        for i in range(len(self.n_hidden)):
            if i == 0:
                layer = BNNLayer(self.n_in, self.n_hidden[i], prior_sd=self.prior_sd, nonlinearity=self.transf)
            else:
                layer = BNNLayer(self.n_hidden[i - 1], self.n_hidden[i], prior_sd=self.prior_sd, nonlinearity=self.transf)
            self.layers.append(layer)
            if i == (len(self.n_hidden) - 1):
                layer = BNNLayer(self.n_hidden[i], self.n_out, prior_sd=self.prior_sd, nonlinearity=self.outf)
                self.layers.append(layer)

    def save_old_params(self):
        for layer in self.layers:
            layer.save_old_params()

    def reset_to_old_params(self):
        for layer in self.layers:
            layer.reset_to_old_params()

    def compression_improvement(self):
        """KL divergence KL[old_param||new_param]"""
        return sum(l.kl_div_old_new() for l in self.layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        return sum(l.kl_div_new_old() for l in self.layers)

    def surprise(self):
        surpr = 0.
        if self.compression:
            surpr += self.compression_improvement()
        if self.information_gain:
            surpr += self.inf_gain()
        return surpr

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        return sum(l.kl_div_new_old() for l in self.layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        return sum(l.kl_div_new_prior() for l in self.layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        return sum(l.kl_div_prior_new() for l in self.layers)

    def _log_prob_normal(self, input, mu=torch.Tensor([0.]), sigma=torch.Tensor([1.])):
        log_normal = - \
            torch.log(sigma) - torch.log(torch.sqrt(2 * PI_tensor)) - \
            square(input - mu) / (2 * square(sigma))
        return torch.sum(log_normal)

    def forward(self, x, **kwargs):
        output = x
        for _, l in enumerate(self.layers):
            output = l(output, **kwargs)
        return output

    def pred_sym(self, input):
        with torch.no_grad():
            return self.forward(input)

    def loss(self, input, target):

        # MC samples.
        _log_p_D_given_w = []
        for _ in range(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, self.likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        return kl / self.n_batches - log_p_D_given_w / self.n_samples

    def loss_last_sample(self, input, target):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """

        # MC samples.
        _log_p_D_given_w = []
        for _ in range(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(sample|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, self.likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        # Calculate loss function.
        # self.kl_div() should be zero when taking second order step
        return self.kl_div() - log_p_D_given_w / self.n_samples

    def pred_fn(self, input):
        return self.pred_sym(input)

    def train_fn(self, input, target):
        self.opt.zero_grad()
        loss = self.loss(input, target)
        loss.backward()
        self.opt.step()

    def second_order_update(self, loss_or_grads, params, oldparams, step_size):
        # TODO - NOT USED
        """Second-order update method for optimizing loss_last_sample, so basically,
        KL term (new params || old params) + NLL of latest sample. The Hessian is
        evaluated at the origin and provides curvature information to make a more
        informed step in the correct descent direction."""
        pass

    def fast_kl_div(self, step_size):
        kl_component = []
        for m in self.modules():
            if isinstance(m, BNNLayer):
                # compute kl for mu
                mu = m.mu.data
                mu_grad = m.mu.grad.data
                rho_old = m.rho_old
                invH = square(torch.log(1 + torch.exp(rho_old)))
                kl_component.append((square(step_size) * square(mu_grad) * invH).sum())

                # compute kl for b_mu
                b_mu = m.b_mu.data
                b_mu_grad = m.b_mu.grad.data
                b_rho_old = m.b_rho_old
                invH = square(torch.log(1 + torch.exp(b_rho_old)))
                kl_component.append((square(step_size) * square(b_mu_grad) * invH).sum())

                # compute kl for rho
                rho = m.rho.data
                rho_grad = m.rho.grad.data
                rho_old = m.rho_old
                H = 2. * (torch.exp(2 * rho)) / square(1. + torch.exp(rho)) / square(torch.log(1. + torch.exp(rho)))
                invH = 1. / H
                kl_component.append((square(step_size) * square(rho_grad) * invH).sum())

                # compute kl for b_rho
                b_rho = m.b_rho.data
                b_rho_grad = m.b_rho.grad.data
                b_rho_old = m.b_rho_old
                H = 2. * (torch.exp(2 * b_rho)) / square(1. + torch.exp(b_rho)) / square(torch.log(1. + torch.exp(b_rho)))
                invH = 1. / H
                kl_component.append((square(step_size) * square(b_rho_grad) * invH).sum())

        return sum(kl_component)

    def compute_fast_kl_div(self, step_size):
        return self.fast_kl_div(step_size)

    def train_update_fn(self, input, target, step_size=None):
        if self.second_order_update:
            assert(step_size is not None)
            self.opt.zero_grad()
            loss = self.loss_last_sample(input, target)
            loss.backward()
            return self.compute_fast_kl_div(step_size)
        else:
            self.opt.zero_grad()
            return self.loss_last_sample(input, target)

    def f_kl_div_closed_form(self):
        return self.surprise()

    def get_param_values(self):
        """Get the parameters to the dynamics.
        This method is included to ensure consistency with TF policies.
        Returns:
            dict: The parameters (in the form of the state dictionary).
        """
        return self.state_dict()

    def set_param_values(self, state_dict):
        """Set the parameters to the dynamics.
        This method is included to ensure consistency with TF policies.
        Args:
            state_dict (dict): State dictionary.
        """
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    pass