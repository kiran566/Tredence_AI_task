import torch


def get_gate_statistics(model):

    statistics = {}

    for name, layer in [
        ("fc1", model.fc1),
        ("fc2", model.fc2),
        ("fc3", model.fc3)
    ]:

        gates = torch.sigmoid(
            layer.gate_scores
        )

        statistics[name] = {
            "min": gates.min().item(),
            "max": gates.max().item(),
            "mean": gates.mean().item(),
            "below_0.01": (gates < 0.01).sum().item(),
            "total": gates.numel()
        }

    return statistics   