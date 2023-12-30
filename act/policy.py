import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.few_shot = args_override["few_shot"]
        self.ent_weight = args_override["ent_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None, **kwargs):
        env_state = None
        # Mean and std from Imagenet => Should we change this?
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad, **kwargs
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1

            # For uninformative baseline
            loss_dict["l1_uninf"] = torch.tensor(0.0)
            if not self.few_shot:
                env_ind = kwargs.get("env_ind")
                skill_ind = torch.zeros(
                    kwargs.get("skill_ind").shape, dtype=torch.float32
                ).cuda()  # uninformative baseline
                a_hat_uninf, _, (_, _) = self.model(
                    qpos,
                    image,
                    env_state,
                    actions,
                    is_pad,
                    skill_ind=skill_ind,
                    env_ind=env_ind,
                )
                all_l1_uninf = F.l1_loss(actions, a_hat_uninf, reduction="none")
                l1_uninf = (all_l1_uninf * ~is_pad.unsqueeze(-1)).mean()
                loss_dict["l1_uninf"] = (
                    self.ent_weight if self.ent_weight else 0.01
                ) * l1_uninf

            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = (
                loss_dict["l1"]
                + loss_dict["l1_uninf"]
                + loss_dict["kl"] * self.kl_weight
            )
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, **kwargs
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
