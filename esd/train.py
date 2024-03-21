import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict
from copy import deepcopy
from tqdm import trange

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, SchedulerMixin


# colab無料版で動作することを確認

def load_models(
        version_name: str
    ) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin]:
    pipe = StableDiffusionPipeline.from_pretrained(version_name)

    # text encoder & tokenizer
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # unet & scheduler
    unet = pipe.unet
    scheduler = pipe.scheduler

    del pipe

    return tokenizer, text_encoder, unet, scheduler

def freeze_unfreeze_params(method: str, unet: UNet2DConditionModel) -> UNet2DConditionModel:
    # freeze all parameters
    for param in unet.parameters():
        param.requires_grad = False
    
    if method in ["ESD-x", "ESD-u"]:
        # ESD-x: cross attention only (attn2)
        # ESD-u: only non-cross attention (except attn2)
        print("update parameters")
        for param_name, module in unet.named_modules():
            # パラメータ名を見てunfreezeする
            if method == "ESD-x":
                if "attn2" in param_name:
                    print(param_name)
                    for param in module.parameters():
                        param.require_grad = True
            else:
                if "attn2" not in param_name:
                    print(param_name)
                    for param in module.parameters():
                        param.require_grad = True
        return unet
    else:
        raise ValueError(f"invlaid method: {method}")


class Trainer:
    def __init__(
            self, 
            method: str,
            version_name: str, 
            erasing_concept: str,
            devices: list[str]
        ) -> None:
        self.tokenizer, self.text_encoder, self.fine_tune_unet, self.scheduler = load_models(version_name)
        
        # copy unet
        self.original_unet = deepcopy(self.fine_tune_unet)
        # freeze original unet
        self.original_unet.eval()

        # unfreeze fine tune unet param 
        self.fine_tune_unet = freeze_unfreeze_params(method, self.fine_tune_unet)

        self.device_list = devices
        self.device = devices[0]
    
        self.erasing_concept = erasing_concept

    
    @torch.no_grad()
    def text_encoding(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_size is None:
            raise ValueError("batch size is not optioned.")
    
        uncond_input_ids = self.tokenizer(
            [""] * self.batch_size, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids

        cond_input_ids = self.tokenizer(
            [self.erasing_concept] * self.batch_size, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids

        self.text_encoder.to(self.device)

        uncond_embeddings = self.text_encoder(uncond_input_ids.to(self.device))[0]
        cond_embeddings = self.text_encoder(cond_input_ids.to(self.device))[0]

        # 推論コードを追加することを見越してCPUに戻すだけにする
        self.text_encoder.cpu()

        return uncond_embeddings, cond_embeddings
    
    def train(
        self,
        save_path: str,
        eta: float=1,
        num_iterations: int=1000,
        batch_size: int=1,
        learning_rate: float=1e-5,
        optimizer_name: str="Adam"
    ) -> None:
        
        self.iterations = num_iterations
        self.batch_size = batch_size
        self.guidance_scale = eta

        optimizer_name = optimizer_name.lower()
        
        # optimizerの設定
        # lr以外のパラメータについては言及がないのでデフォルトを使用
        if optimizer_name == "adam":
            optimizer = optim.Adam(
                self.fine_tune_unet.parameters(),
                lr=learning_rate
            )
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                self.fine_tune_unet.parameters(),
                lr=learning_rate
            )
        else:
            raise ValueError(f"invalid optimizer: {optimizer_name}")
    
        uncond_text_embeddings, cond_text_embeddings = self.text_encoding()
        text_embeddings = torch.cat([uncond_text_embeddings, cond_text_embeddings]).to(self.device)

        # GPUが2枚以上ならdevice_list[0]にfine-tune-unetを乗せてdevice_list[1]にoriginal-unetを乗せる
        if len(self.device_list) > 1:
            self.original_unet.to(self.device_list[1])
            self.fine_tune_unet.to(self.device)

        pbar = trange(0, self.iterations, desc="Iteration")
        for _ in pbar:
            optimizer.zero_grad()

            # Time step sampled uniformly
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (1, ))
            t = t.long()
            t = t.to(self.device)
            
            noise = torch.randn(
                (self.batch_size, self.unet.config.in_channels, 512 // 8, 512 // 8)
            ).repeat(1, 1, 1, 1).to(self.device)
            latents = noise * self.scheduler.init_noise_sigma
            latents = torch.cat([latents] * 2).to(self.device)

            noisy_latents = self.scheduler.add_noise(latents, noise, t)

            # calc εθ*(xt, t) − η[εθ*(xt, c, t) − εθ*(xt, t)]
            # GPUが1つの場合はここでCPUからGPUに乗せて推論したらCPUに戻す
            if len(self.device_list) == 1:
                self.original_unet.to(self.device)
            with torch.no_grad():
                model_pred = self.original_unet(
                    noisy_latents.to(self.original_unet.device), 
                    t.to(self.original_unet.device), 
                    text_embeddings.to(self.original_unet.device)
                ).sample
                noise_prediction_uncond, noise_prediction_text = model_pred.chunk(2)
                original_pred = noise_prediction_uncond - self.guidance_scale * (noise_prediction_text - noise_prediction_uncond)
            
            if len(self.device_list) == 1:
                self.original_unet.cpu()
            else:
                original_pred.to(self.fine_tune_unet.device)

            # εθ(xt, c, t)
            # 先ほどと同様にGPUが1枚ならここでCPUからGPUに乗せてoptimizer.stepしたらCPUに戻す
            if len(self.device_list) == 1:
                self.fine_tune_unet.to(self.device)
            latents = noise * self.scheduler.init_noise_sigma
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            fine_tune_pred = self.fine_tune_unet(noisy_latents, t, cond_text_embeddings).sample

            # loss
            loss = F.mse_loss(fine_tune_pred, original_pred, reduction="mean")
            loss.backward()
            optimizer.step()

            if len(self.device_list) == 1:
                self.fine_tune_unet.cpu()
            
            pbar.set_postfix(OrderedDict(loss=loss.detach().item()))
        
        # モデルの保存
        self.fine_tune_unet.eval()
        self.fine_tune_unet.save_pretrained(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concept", type=str, required=True, help="the name of the concept you want to erase.")
    parser.add_argument("--method", type=str, choices=["ESD-x", "ESD-u"], help="erasing method. ESD-x is updating only cross attention and ESD-u is updating only non-cross-attention.")
    parser.add_argument("--sd_version", type=str, default="compvis/stable-diffusion-v1-4", help="model name.")
    
    parser.add_argument("--save_dir", type=str, default="fine-tuned", help="path to save directory")
    parser.add_argument("--eta", type=float, default=1.0, help="negative guidance. the value is 1.0 in the paper.")
    parser.add_argument("--iter", type=int, default=1000, help="the number of iteration. the value is 1000 in the paper.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size. the value is 1 in the paper.")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate. the value is 1e-5 in the paper.")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer name. the name is adam in the paper.")

    args = parser.parse_args()

    trainer = Trainer(
        method=args.method,
        version_name=args.sd_version,
        erasing_concept=args.target_concept,
        device="cuda:1"
    )

    trainer.train(
        args.save_dir,
        args.eta,
        args.iter,
        args.batch_size,
        args.lr,
        args.optim
    )
