from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from inspect import signature
from typing import Any, Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import lightning.pytorch as pl
import numpy as np
import optax
import pyro
import torch
import torchmetrics.functional as tmf
from lightning.pytorch.strategies.ddp import DDPStrategy
from pyro.nn import PyroModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networkvi import REGISTRY_KEYS
from networkvi.module.base import (
    BaseModuleClass,
    LossOutput,
)
from networkvi.train._constants import METRIC_KEYS
from networkvi.module import Classifier

from ._metrics import ElboMetric

JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

def _compute_kl_weight(
    epoch: int,
    step: int,
    n_epochs_kl_warmup: Optional[int],
    n_steps_kl_warmup: Optional[int],
    max_kl_weight: float = 1.0,
    min_kl_weight: float = 0.0,
) -> float:
    """Computes the kl weight for the current step or epoch.

    If both `n_epochs_kl_warmup` and `n_steps_kl_warmup` are None `max_kl_weight` is returned.

    Parameters
    ----------
    epoch
        Current epoch.
    step
        Current step.
    n_epochs_kl_warmup
        Number of training epochs to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    """
    if min_kl_weight > max_kl_weight:
        raise ValueError(
            f"min_kl_weight={min_kl_weight} is larger than max_kl_weight={max_kl_weight}."
        )

    slope = max_kl_weight - min_kl_weight
    if n_epochs_kl_warmup:
        if epoch < n_epochs_kl_warmup:
            return slope * (epoch / n_epochs_kl_warmup) + min_kl_weight
    elif n_steps_kl_warmup:
        if step < n_steps_kl_warmup:
            return slope * (step / n_steps_kl_warmup) + min_kl_weight
    return max_kl_weight


class TrainingPlan(pl.LightningModule):
    """Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.

    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/dev/model_user_guide`.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`. Only activated when `n_epochs_kl_warmup` is
        set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from `min_kl_weight` to
        `max_kl_weight`. Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed.
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        max_kl_weight: float = 1.0,
        min_kl_weight: float = 0.0,
        **loss_kwargs,
    ):
        super().__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs
        self.min_kl_weight = min_kl_weight
        self.max_kl_weight = max_kl_weight
        self.optimizer_creator = optimizer_creator

        if self.optimizer_name == "Custom" and self.optimizer_creator is None:
            raise ValueError("If optimizer is 'Custom', `optimizer_creator` must be provided.")

        self._n_obs_training = None
        self._n_obs_validation = None

        # automatic handling of kl weight
        self._loss_args = set(signature(self.module.loss).parameters.keys())
        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        self.initialize_train_metrics()
        self.initialize_val_metrics()

    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: Optional[int] = None):
        """Initialize ELBO metric and the metric collection."""
        rec_loss = ElboMetric("reconstruction_loss", mode, "obs")
        rec_loss_expression = ElboMetric("reconstruction_loss_expression", mode, "obs")
        rec_loss_accessibility = ElboMetric("reconstruction_loss_accessibility", mode, "obs")
        rec_loss_protein = ElboMetric("reconstruction_loss_protein", mode, "obs")
        kl_local = ElboMetric("kl_local", mode, "obs")
        kl_local_z = ElboMetric("kl_divergence_z", mode, "obs")
        kl_local_paired = ElboMetric("kl_divergence_paired", mode, "obs")
        kl_global = ElboMetric("kl_global", mode, "batch")
        # n_total can be 0 if there is no validation set, this won't ever be used
        # in that case anyway
        n = 1 if n_total is None or n_total < 1 else n_total
        elbo = rec_loss + kl_local + (1 / n) * kl_global
        elbo.name = f"elbo_{mode}"
        collection = OrderedDict(
            [(metric.name, metric) for metric in [elbo, rec_loss, rec_loss_expression, rec_loss_accessibility, rec_loss_protein, kl_local, kl_local_z, kl_local_paired, kl_global]]
        )
        """
        collection = OrderedDict(
            [(metric.name, metric) for metric in [elbo, rec_loss, rec_loss_expression, kl_local, kl_global]]
        )
        """
        #return elbo, rec_loss, kl_local, kl_global, collection
        return elbo, rec_loss, rec_loss_expression, rec_loss_accessibility, rec_loss_protein, kl_local, kl_local_z, kl_local_paired, kl_global, collection

    def initialize_train_metrics(self):
        """Initialize train related metrics."""
        """
        (
            self.elbo_train,
            self.rec_loss_train,
            self.kl_local_train,
            self.kl_global_train,
            self.train_metrics,
        ) = self._create_elbo_metric_components(mode="train", n_total=self.n_obs_training)
        """
        (
            self.elbo_train,
            self.rec_loss_train,
            self.rec_loss_train_expression,
            self.rec_loss_train_accessibility,
            self.rec_loss_train_protein,
            self.kl_local_train,
            self.kl_local_train_z,
            self.kl_local_train_paired,
            self.kl_global_train,
            self.train_metrics,
        ) = self._create_elbo_metric_components(mode="train", n_total=self.n_obs_training)
        self.elbo_train.reset()

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        (
            self.elbo_val,
            self.rec_loss_val,
            self.rec_loss_val_expression,
            self.rec_loss_val_accessibility,
            self.rec_loss_val_protein,
            self.kl_local_val,
            self.kl_local_val_z,
            self.kl_local_val_paired,
            self.kl_global_val,
            self.val_metrics,
        ) = self._create_elbo_metric_components(mode="validation", n_total=self.n_obs_validation)
        self.elbo_val.reset()

    @property
    def use_sync_dist(self):
        return isinstance(self.trainer.strategy, DDPStrategy)

    @property
    def n_obs_training(self):
        """Number of observations in the training set.

        This will update the loss kwargs for loss rescaling.

        Notes
        -----
        This can get set after initialization
        """
        return self._n_obs_training

    @n_obs_training.setter
    def n_obs_training(self, n_obs: int):
        if "n_obs" in self._loss_args:
            self.loss_kwargs.update({"n_obs": n_obs})
        self._n_obs_training = n_obs
        self.initialize_train_metrics()

    @property
    def n_obs_validation(self):
        """Number of observations in the validation set.

        This will update the loss kwargs for loss rescaling.

        Notes
        -----
        This can get set after initialization
        """
        return self._n_obs_validation

    @n_obs_validation.setter
    def n_obs_validation(self, n_obs: int):
        self._n_obs_validation = n_obs
        self.initialize_val_metrics()

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(*args, **kwargs)

    @torch.inference_mode()
    def compute_and_log_metrics(
        self,
        loss_output: LossOutput,
        metrics: dict[str, ElboMetric],
        mode: str,
    ):
        """Computes and logs metrics.

        Parameters
        ----------
        loss_output
            LossOutput object from networkvi-tools module
        metrics
            Dictionary of metrics to update
        mode
            Postfix string to add to the metric name of
            extra metrics
        """
        rec_loss = loss_output.reconstruction_loss_sum
        rec_loss_expression = loss_output.reconstruction_loss["reconstruction_loss_expression"].sum() if "reconstruction_loss_protein" in loss_output.reconstruction_loss.keys() else 0
        rec_loss_accessibility = loss_output.reconstruction_loss["reconstruction_loss_accessibility"].sum() if "reconstruction_loss_protein" in loss_output.reconstruction_loss.keys() else 0
        rec_loss_protein = loss_output.reconstruction_loss["reconstruction_loss_protein"].sum()  if "reconstruction_loss_protein" in loss_output.reconstruction_loss.keys() else 0

        n_obs_minibatch = loss_output.n_obs_minibatch
        kl_local = loss_output.kl_local_sum
        kl_local_z = loss_output.kl_local["kl_divergence_z"].sum()  if "kl_divergence_z" in loss_output.kl_local.keys() else 0
        kl_local_paired = loss_output.kl_local["kl_divergence_paired"].sum()  if "kl_divergence_paired" in loss_output.kl_local.keys() else 0
        kl_global = loss_output.kl_global_sum

        # Use the torchmetric object for the ELBO
        # We only need to update the ELBO metric
        # As it's defined as a sum of the other metrics

        metrics[f"elbo_{mode}"].update(
            reconstruction_loss=rec_loss,
            kl_local=kl_local,
            kl_global=kl_global,
            n_obs_minibatch=n_obs_minibatch,
        )

        for metric_name in ["reconstruction_loss_expression", "reconstruction_loss_accessibility", "reconstruction_loss_protein", "kl_divergence_z", "kl_divergence_paired"]:
            #metric_name = "reconstruction_loss_accessibility"
            metrics[f"{metric_name}_{mode}"].update(
                reconstruction_loss=rec_loss,
                reconstruction_loss_expression=rec_loss_expression,
                reconstruction_loss_accessibility=rec_loss_accessibility,
                reconstruction_loss_protein=rec_loss_protein,
                kl_local=kl_local,
                kl_divergence_z=kl_local_z,
                kl_divergence_paired=kl_local_paired,
                kl_global=kl_global,
                n_obs_minibatch=n_obs_minibatch,
            )

        # pytorch lightning handles everything with the torchmetric object
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=n_obs_minibatch,
            sync_dist=self.use_sync_dist,
        )

        # accumlate extra metrics passed to loss recorder
        for key in loss_output.extra_metrics_keys:
            met = loss_output.extra_metrics[key]
            if isinstance(met, torch.Tensor):
                if met.shape != torch.Size([]):
                    raise ValueError("Extra tracked metrics should be 0-d tensors.")
                met = met.detach()
            self.log(
                f"{key}_{mode}",
                met,
                on_step=False,
                on_epoch=True,
                batch_size=n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log(
            "train_loss",
            scvi_loss.loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return scvi_loss.loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        # loss kwargs here contains `n_obs` equal to n_training_obs
        # so when relevant, the actual loss value is rescaled to number
        # of training examples
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log(
            "validation_loss",
            scvi_loss.loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

    def _optimizer_creator_fn(self, optimizer_cls: Union[torch.optim.Adam, torch.optim.AdamW]):
        """Create optimizer for the model.

        This type of function can be passed as the `optimizer_creator`
        """
        return lambda params: optimizer_cls(
            params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )

    def get_optimizer_creator(self):
        """Get optimizer creator for the model."""
        if self.optimizer_name == "Adam":
            optim_creator = self._optimizer_creator_fn(torch.optim.Adam)
        elif self.optimizer_name == "AdamW":
            optim_creator = self._optimizer_creator_fn(torch.optim.AdamW)
        elif self.optimizer_name == "Custom":
            optim_creator = self.optimizer_creator
        else:
            raise ValueError("Optimizer not understood.")

        return optim_creator

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer = self.get_optimizer_creator()(params)
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        return config

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        return _compute_kl_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_kl_warmup,
            self.n_steps_kl_warmup,
            self.max_kl_weight,
            self.min_kl_weight,
        )


class AdversarialTrainingPlan(TrainingPlan):
    """Train vaes with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
    scale_adversarial_loss
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following opposite of
        kl warmup.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = False,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        if adversarial_classifier is True:
            self.n_output_classifier = self.module.n_batch
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
        else:
            self.adversarial_classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def loss_adversarial_classifier(self, z, batch_index, predict_true_class=True):
        """Loss for adversarial classifier."""
        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))

        if predict_true_class:
            cls_target = torch.nn.functional.one_hot(batch_index.squeeze(-1), n_classes)
        else:
            one_hot_batch = torch.nn.functional.one_hot(batch_index.squeeze(-1), n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss = scvi_loss.loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
            loss += fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1
