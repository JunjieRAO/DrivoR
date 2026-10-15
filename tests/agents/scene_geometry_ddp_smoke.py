"""Run with torchrun --standalone --nproc_per_node=2 (CPU/Gloo, two steps)."""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_scene_geometry import config, batch_targets
from navsim.agents.drivoR.scene_geometry.geometry_encoder import SceneGeometryEncoder, GeometryAdapter
from navsim.agents.drivoR.scene_geometry.geometry_loss import geometry_loss


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SceneGeometryEncoder(config(), 8)
        self.adapter = GeometryAdapter(config())
    def forward(self, memory, targets):
        predictions = self.encoder(memory)
        output = self.adapter(memory, predictions)
        return geometry_loss(predictions, targets)["geometry/loss"] + output.square().mean()


if __name__ == "__main__":
    dist.init_process_group("gloo")
    torch.set_num_threads(1)
    model = DistributedDataParallel(Model(), find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # Different positive counts on ranks, including an entirely empty/missing-map sample.
    targets = batch_targets(config(), empty=dist.get_rank() == 1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(1, 8, 32), targets)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    if dist.get_rank() == 0:
        print("DDP_GEOMETRY_OK: two ranks, unequal positive counts, two optimizer steps")
    dist.destroy_process_group()
