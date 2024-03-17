import os
import argparse
import PIL
import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

class Trainer():
    def __init__(self, lr=5*(1e-3), N=5, it=1500, seed=0, device="cuda"):
        self.lr = lr
        self.N = N
        self.it = it
        self.device = device
        self.seed = seed

    def load_model(self, model_path):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(self.device)
        self.noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.weight_dtype = torch.float32
    
    def load_image(self, image_path):
        input_image = Image.open(image_path).convert("RGB")
        image_transform = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )
        init_image = image_transform(input_image)
        init_image = init_image[None].to(self.device, self.weight_dtype)

        with torch.no_grad():
            init_latents = self.vae.encode(init_image).latent_dist.sample()
            self.init_latents = init_latents * 0.18215
    
    def train(self, source_prompt, optimizer_name="adam"):
        optimizer_name = optimizer_name.lower()
        torch.manual_seed(self.seed)
        text_ids = self.tokenizer(
            source_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_ids = text_ids.to(self.device)

        with torch.no_grad():
            self.source_embedding = self.text_encoder(text_ids)[0]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        target_embedding = self.source_embedding.float()
        self.src_emb = target_embedding[:, :77-self.N].detach().clone()
        optimized_embedding = target_embedding[:, 77-self.N:].detach().clone()
        
        optimized_embedding.requires_grad = True
        # Optimizer
        if optimizer_name == "adam":
            optimizer = optim.Adam([optimized_embedding], lr=self.lr,)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW([optimized_embedding], lr=self.lr,)
        elif optimizer_name == "adagrad":
            optimizer = optim.Adagrad([optimized_embedding], lr=self.lr,)
        elif optimizer_name == "adadelta":
            optimizer = optim.Adadelta([optimized_embedding], lr=self.lr,)
        else:
            raise ValueError("unvalid optimizer name.")

        pbar = tqdm(range(self.it))
        for _ in pbar:
            noise = torch.randn_like(self.init_latents, device=self.device)
            bsz = self.init_latents.shape[0]

            # sample a random timestep for a image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (forward process)
            noisy_latents = self.noise_scheduler.add_noise(self.init_latents, noise, timesteps).to(self.device)
            cond_embedding = torch.cat([self.src_emb, optimized_embedding], dim=1).to(self.device)
            
            noise_pred = self.unet(noisy_latents, timesteps, cond_embedding).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({"loss": loss.item()})

        # Acoording to the paper, the HiPer embedding is calibrated 0.8 (const value) before concatenating it with target embedding.
        optimized_embedding.requires_grad = False
        self.optim_emb = optimized_embedding.detach().clone() * 0.8 

    
    def infer(self, target_prompt, num_inference_steps, guidance_scale):
        torch.manual_seed(self.seed)
        print(f"seed : {self.seed}")
        print(f"target prompt : {target_prompt}")
        print(f"num inference : {num_inference_steps}")
        print(f"guidance scale : {guidance_scale}")

        text_ids = self.tokenizer(
            target_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_ids = text_ids.to(self.device)

        with torch.no_grad():
            target_embedding = self.text_encoder(text_ids)[0]
        
        emb_comp = torch.cat([target_embedding[:, :77-self.N], self.optim_emb], dim=1).to(self.device)
        emb_comp = torch.cat([emb_comp] * 2).to(self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        try:
            latents = torch.randn_like(self.init_latents, device=self.device)
        except AttributeError:
            latents = torch.randn((1, 4, 64, 64), device=self.device)
        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order

        with tqdm(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2).to(self.device)
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=emb_comp).sample

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    pbar.update()
        
        output_image = self.decode_latents(latents)
        output_image = self.numpy_to_pil(output_image)
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        for i in range(len(output_image)):
            output_image[i].save(f"{now}-output{i + 1}.jpg")
    
    def save_optim_emb(self, out_dir=""):
        torch.save(self.optim_emb.detach().cpu(), os.path.join(out_dir, "optim_emb.pt"))
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        return image
    
    def numpy_to_pil(self, images) -> PIL.Image:
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def pipeline(self, model_name, image_path, source_prompt, target_prompt, num_inference_steps=200, guidance_scale=7):
        self.load_model(model_name)
        self.load_image(image_path)
        self.train(source_prompt)
        self.save_optim_emb()
        self.infer(target_prompt, num_inference_steps, guidance_scale)
    
    def only_infer(self, model_name, target_prompt, pt_path, num_inference_steps=200, guidance_scale=14):
        self.optim_emb = torch.load(pt_path, map_location="cpu").to(self.device)
        self.load_model(model_name)
        self.infer(target_prompt, num_inference_steps, guidance_scale)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance", type=float, default=18.0)
    parser.add_argument("--infer", action='store_true') # default is False
    parser.add_argument("--seed", type=int, default=1873282)
    parser.add_argument("--num_steps", type=int, default=100, help="inference step when generating image")
    parser.add_argument("--optim", type=str, default="adam", help="choose optimizer", choices=["adam", "adamw", "adagrad", "adadelta"])
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    lr = 5 * (1e-3) # optimizer learning rate
    N = args.tokens # optimizing last N tokens, recommended N=5 in the paper
    it = args.iters # 1000steps for optimizing
    num_inference_step = args.num_steps # inference steps for image generation
    guidance_scale = args.guidance # guidance scale (classifier free guidance) for image generation
    model_name = "CompVis/stable-diffusion-v1-4" # model name
    source_text_prompt = "A sitting dog" # image prompt
    target_text_prompt = "A jumping dog" # target prompt for image generation
    source_image_path = "sample-512x512.jpg" # source image path
    trainer = Trainer(N=N, it=it, lr=lr, seed=args.seed)

    if args.infer:
        trainer.only_infer(model_name, target_text_prompt, "optim_emb.pt", num_inference_step, guidance_scale)
    else:
        trainer.pipeline(model_name, source_image_path, source_text_prompt, target_text_prompt, num_inference_step, guidance_scale)

if __name__ == "__main__":
    main()