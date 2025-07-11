import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from collections import namedtuple
from typing import Optional
from math import isclose

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from ..trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)


def apply_temperature(probabilities: list[float], temperature: float) -> list[float]:
    """
    Apply temperature scaling to a list of probabilities using PyTorch.

    Args:
        probabilities (list[float]): Initial probability distribution
        temperature (float): Temperature parameter (> 0)

    Returns:
        list[float]: Scaled and normalized probabilities
    """
    probs_tensor = t.tensor(probabilities, dtype=t.float32)
    logits = t.log(probs_tensor)
    scaled_logits = logits / temperature
    scaled_probs = t.nn.functional.softmax(scaled_logits, dim=0)

    return scaled_probs.tolist()

# ================================================================
# NDropout Batch Top-K SAE
# ================================================================

class SeqDropoutBatchTopKTrainer(SAETrainer):
    """Sequential-Dropout Batch Top-K SAE.

    We iterate over the ordered top-k activations (top-1, top-2, …, top-k).
    In iteration *i* (1-indexed), the reconstruction uses the top-*i* features,
    **but a stop-gradient is applied to the first *(i − 1)* features** so only the
    newly-added feature receives gradient signal.  This forces each individual
    feature to learn to pick up unique signal beyond what the previously learned
    features already explain (nested dropout objective in the OpenAI April-update).
    """

    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = MatryoshkaBatchTopKSAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "SeqDropoutBatchTopKSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Auto-encoder – we reuse the Matryoshka implementation with a single group.
        self.ae = dict_class(activation_dim, dict_size, k, [dict_size])

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        # Learning-rate: scaling law from paper unless user overrides.
        if lr is not None:
            self.lr = lr
        else:
            scale = dict_size / (2 ** 14)
            self.lr = 2e-4 / scale ** 0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # heuristic from paper B.1

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=self.device)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

    # ---------------------------------------------------------------------
    # Auxiliary loss (identical to Matryoshka version, sans bias in decoder)
    # ---------------------------------------------------------------------
    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features == 0:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

        k_aux = min(self.top_k_aux, self.dead_features)
        auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)
        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

        auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
        auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

        # Decoder without bias.
        x_reconstruct_aux = auxk_acts_BF @ self.ae.W_dec
        l2_loss_aux = (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
        self.pre_norm_auxk_loss = l2_loss_aux

        # Variance normalisation.
        residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
        loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
        normalized_auxk_loss = l2_loss_aux / loss_denom
        return normalized_auxk_loss.nan_to_num(0.0)

    # ---------------------------------------------------------------------
    # Threshold update (same as Matryoshka)
    # ---------------------------------------------------------------------
    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]
            min_activation = active.min().detach().to(dtype=t.float32) if active.size(0) > 0 else 0.0
            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    # ---------------------------------------------------------------------
    # Main loss – nested-dropout accumulation over top-1 .. top-k features.
    # ---------------------------------------------------------------------
    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )

        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)

        # Sort the k active features per example by activation magnitude (descending).
        vals_BK, idx_BK = post_relu_acts_BF.topk(self.k, dim=1, sorted=True)

        # Sequential-dropout reconstruction with stop-gradient.
        prev_detached = t.zeros_like(f)
        l2_losses = []

        for i in range(self.k):
            acts_new, prev_detached = self.add_feature_with_stop_gradient(
                prev_detached, idx_BK[:, i : i + 1], vals_BK[:, i : i + 1]
            )

            # Reconstruction uses frozen previous activations + current live activation.
            x_reconstruct = (
                self.ae.b_dec
                + prev_detached @ self.ae.W_dec  # no gradient
                + acts_new @ self.ae.W_dec  # gradient flows only through new feature
            )

            l2_loss_i = (x - x_reconstruct).pow(2).sum(dim=-1).mean()
            l2_losses.append(l2_loss_i)

        l2_losses_stack = t.stack(l2_losses)
        mean_l2_loss = l2_losses_stack.mean()
        min_l2_loss = l2_losses_stack.min().item()
        max_l2_loss = l2_losses_stack.max().item()

        self.effective_l0 = self.k

        # Update dead-feature counters.
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        auxk_loss = self.get_auxiliary_loss((x - x_reconstruct).detach(), post_relu_acts_BF)
        loss = mean_l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_reconstruct,
                f,
                {
                    "l2_loss": mean_l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "min_l2_loss": min_l2_loss,
                    "max_l2_loss": max_l2_loss,
                },
            )

    # ---------------------------------------------------------------------
    # Training update step (unchanged except for W_dec normalisation)
    # ---------------------------------------------------------------------
    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # Remove gradient component parallel to decoder directions.
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        ).T
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        # Re-normalise decoder rows to unit norm using helper util
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()

    def pairwise_distances(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        """
        Computes pairwise distances between two tensors.
        x: tensor of shape (n, d)
        y: tensor of shape (m, d)
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * t.mm(x, y.t())
        return t.sqrt(t.clamp(dist, min=0.0))

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
    
    # def geometric_median(self, x: t.Tensor, max_iter: int = 100, tol: float = 1e-5) -> t.Tensor:
    #     """
    #     Computes the geometric median of a tensor.
    #     x: tensor of shape (n, d)
    #     """
    #     median = x.mean(dim=0)
    #     for _ in range(max_iter):
    #         prev_median = median.clone()
    #         distances = self.pairwise_distances(x, median.unsqueeze(0)).squeeze()
    #         weights = 1.0 / t.clamp(distances, min=1e-6)
    #         weights_sum = weights.sum()
    #         median = (weights.unsqueeze(1) * x).sum(dim=0) / weights_sum
    #         if t.norm(median - prev_median) < tol:
    #             break
    #     return median

    # ---------------------------------------------------------------------
    # Config dump for logging / checkpoint.
    # ---------------------------------------------------------------------
    @property
    def config(self):
        return {
            "trainer_class": "SeqDropoutBatchTopKTrainer",
            "dict_class": "MatryoshkaBatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    def add_feature_with_stop_gradient(self, prev_detached: t.Tensor, idx_new: t.Tensor, val_new: t.Tensor):
        """Build sparse activation tensor for *idx_new* while keeping *prev_detached* frozen.

        Parameters
        ----------
        prev_detached : torch.Tensor  (B, F)
            Previously selected features with *requires_grad=False*.
        idx_new : torch.Tensor  (B, 1)
            Index of the newly-added feature for each example in the batch.
        val_new : torch.Tensor  (B, 1)
            Activation value of the newly-added feature for each example.

        Returns
        -------
        acts_new : torch.Tensor  (B, F)
            Sparse tensor with only the new feature active **with gradients enabled**.
        prev_detached_new : torch.Tensor  (B, F)
            Updated detached tensor that now also includes the new feature (detached).
        """
        acts_new = t.zeros_like(prev_detached)
        acts_new.scatter_(1, idx_new, val_new)
        prev_detached_new = prev_detached + acts_new.detach()
        return acts_new, prev_detached_new
