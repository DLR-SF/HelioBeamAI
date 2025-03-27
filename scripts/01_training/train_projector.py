# scripts/train_projection_net.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from HelioBeamAI.model.beam_projector import FocalSpotToDirectionsNet, RayTracingUtils

def prepare_inputs(cond_data, targets_all, device):
    eval_points = cond_data['evalPoints_all'].to(device)
    heliostat_pos = cond_data['hp_all'].to(device)

    directions = (eval_points[:, 0] - heliostat_pos).float()
    lengths = torch.norm(directions, dim=1, keepdim=True)
    normalized_directions = directions / lengths
    normalized_lengths = lengths / lengths.max()
    inputs_nn = torch.cat([normalized_directions, normalized_lengths], dim=-1).float()

    targets_all = targets_all.permute(0, 1, 3, 2).to(device)

    return eval_points, inputs_nn, normalized_directions, targets_all

def forward_step(model, focal_spot, heliostat_input):
    return model(focal_spot.unsqueeze(0).unsqueeze(0), heliostat_input.unsqueeze(0)).squeeze(0)

def train_projection_net(model, targets_all, cond_data, num_epochs=100, learning_rate=1e-4, device='cpu', log_dir='./logs', grad_accum_steps=4):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    writer = SummaryWriter(log_dir)

    eval_points, inputs_nn_all, norm_dirs_all, targets_all = prepare_inputs(cond_data, targets_all, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_weights = 1 / torch.arange(1, 6, dtype=torch.float32, device=device)

    step, scale = 0, 4
    losses_all = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(targets_all.size(0)), desc=f"Epoch {epoch+1}/{num_epochs}"):
            focal_spot = targets_all[i, 0]
            target_diffs = targets_all[i, 1:]
            heliostat_input = inputs_nn_all[i]
            ap_diffs = eval_points[i][1:]

            if scale > 1:
                focal_spot = F.interpolate(focal_spot.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze()

            pred_dev = forward_step(model, focal_spot, heliostat_input)
            pred_dir = F.normalize(norm_dirs_all[i].unsqueeze(-1).unsqueeze(-1) + pred_dev * 1e-2, dim=0).permute(1, 2, 0)

            origins = RayTracingUtils.get_origins(device, bins=focal_spot.size(0))
            directions = pred_dir.view(-1, 3)
            energies = focal_spot.flatten()

            projections = [RayTracingUtils.ray_plane_intersection(directions, origins, energies, ap.to(device), num_bins1=focal_spot.size(-1)) for ap in ap_diffs]
            if scale > 1:
                projections = [F.avg_pool2d(p.unsqueeze(0).unsqueeze(0), kernel_size=scale).squeeze() for p in projections]

            losses = torch.stack([RayTracingUtils.calc_flux_deviation(t, p) for t, p in zip(target_diffs, projections)])
            loss = (losses * loss_weights).mean()

            loss.backward()
            if (i + 1) % grad_accum_steps == 0 or i == targets_all.size(0) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            losses_all.append(loss.item())

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)
            writer.add_scalar("Grad Accum Steps", grad_accum_steps, step)
            step += 1

        epoch_loss = total_loss / targets_all.size(0)
        writer.add_scalar("Epoch Loss/train", epoch_loss, epoch + 1)
        print(f"Epoch {epoch+1} completed - Avg Loss: {100 * epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    writer.close()
    save_best_model(model, losses_all, cond_data['focal_dist_all'], scale)

def save_best_model(model, losses_all, focal_dist_all, scale):
    model_dir = './ProjectionEval'
    os.makedirs(model_dir, exist_ok=True)

    model_files = [f for f in os.listdir(model_dir) if f.startswith('focal_to_directions_model_s') and f.endswith('.pt')]
    best_loss = float('inf')
    if model_files:
        best_file = sorted(model_files)[-1]
        best_loss = float(best_file.split('_')[-1].split('.pt')[0])
        print(f"Loaded best loss: {best_loss:.4f}")

    final_loss = float(np.mean(losses_all))
    if final_loss < best_loss:
        path = f'{model_dir}/focal_to_directions_model_s{scale}_{final_loss:.4f}.pt'
        torch.save(model.state_dict(), path)
        print(f"Model saved: {path}")

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    targets = torch.load('./ProjectionEval/targets_all.pt').to(device)
    cond = torch.load('./ProjectionEval/cond_data.pt')

    model = FocalSpotToDirectionsNet()
    train_projection_net(model, targets, cond, num_epochs=300, learning_rate=1e-3, device=device)
