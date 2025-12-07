import os
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from peft import LoraConfig, get_peft_model, TaskType, LoraModel

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.0, 
    bias="none", 
    task_type=TaskType.FEATURE_EXTRACTION,  # or SEQ_2_SEQ_LM; doesn't matter much here 
    target_modules=[ 
        "linear1", 
        "linear2", 
        "self_attn.out_proj", 
        "multihead_attn.out_proj", 
    ],
)

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, _ = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = None
        self.kl_weight = args_override['kl_weight']
        self.use_lora = args_override.get("use_lora", True)
        print(f'KL Weight {self.kl_weight}')

        # 1) Optionally load full pretrained weights
        if args_override.get("load_pretrain", False):
            ckpt_path = "/home/kelly_lucy/Desktop/manipulation_project/Manipulation-Final-Project/src/checkpoints/DL_pretrained_test/policy_best.ckpt"
            state = torch.load(ckpt_path)

            # 🔥 STRIP "model." PREFIX
            fixed_state = {}
            for k, v in state.items():
                if k.startswith("model."):
                    fixed_state[k.replace("model.", "", 1)] = v
                else:
                    fixed_state[k] = v

            loading_status = self.model.load_state_dict(fixed_state, strict=False)
            print("Loaded pretrained policy:", loading_status)

            # Save checkpoint BEFORE applying LoRA (pure pretrained base model)
            ckpt_dir = args_override.get("ckpt_dir", None)
            seed = args_override.get("seed", 0)
            if ckpt_dir is not None:
                pre_lora_base_ckpt = os.path.join(
                    ckpt_dir, f"policy_pre_lora_nolora_seed_{seed}.ckpt"
                )
                torch.save(self.model.state_dict(), pre_lora_base_ckpt)
                print(f"Saved pre-LoRA (no LoRA) checkpoint to {pre_lora_base_ckpt}")

        # 2) Optionally apply LoRA on top of the *loaded* transformer
        if self.use_lora:
            self.apply_lora_to_transformer()
            # 3) Freeze everything except LoRA params (and optionally some extras)
            self.freeze_all_except_lora()
            self.build_optimizer(args_override)

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
    def deserialize(self, model_dict):
        return self.model.load_state_dict(model_dict)

    def apply_lora_to_transformer(self):
        # Wrap the existing, already-loaded transformer
        self.model.transformer = LoraModel(self.model.transformer, lora_config, "default")

    def freeze_all_except_lora(self):
        # Freeze everything
        for name, p in self.model.named_parameters():
            p.requires_grad = False

        # LoRA params stay trainable
        for name, p in self.model.transformer.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Trainable params in full model:", n_parameters)

    def build_optimizer(self, args_override):
        lr = args_override['lr']
        lr_backbone = args_override['lr_backbone']
        weight_decay = args_override.get('weight_decay', 1e-4)

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters()
                        if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                        weight_decay=weight_decay)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
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
