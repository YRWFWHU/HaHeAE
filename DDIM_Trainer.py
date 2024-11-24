from diffusers import DDIMScheduler
from tqdm import tqdm
from Unet import Unet
from GCNSemEncoder import semanticEncoder
from HaHeForecasting import HaHeForecasting
import argparse
import torch

def train(args):
    device = torch.device(args.device)

    unet = Unet().to(device)
    semEncoder = semanticEncoder().to(device)
    forcaster = HaHeForecasting().to(device)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    optimizer = torch.optim.Adam(list(unet.parameters()) + list(semEncoder.parameters()) + list(forcaster.parameters()), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    batch = torch.randn(args.batch_size, 9, 40, device=device)
    label = torch.randn(args.batch_size, 9, 3, device=device)
    for i in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        latent_representation, semantic_representation = semEncoder(batch)
        future_prediction = forcaster(latent_representation)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (args.batch_size, 1), device=device)
        noise = torch.randn_like(batch, device=device)
        H_t = scheduler.add_noise(batch, noise, timesteps)
        noise_pred = unet(H_t, timesteps, semantic_representation)
        loss = loss_fn(noise_pred, noise) + loss_fn(future_prediction, label) # dimension mismatch
        loss.backward()
        optimizer.step()


def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="DDIM Trainer")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument("--epochs", type=int, default=130, help="Number of epochs")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--prediction_length", type=int, default=3, help="Prediction length")
        parser.add_argument("--device", type=str, default="cuda:1", help="Device")
        return parser.parse_args()

    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()