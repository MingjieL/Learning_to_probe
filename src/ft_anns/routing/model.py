"""Raw-query router architecture."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ft_anns.config import RouterConfig


class RawQueryRouter(nn.Module):
    """Map raw query vectors directly to Fixed Partition bucket logits."""

    def __init__(
        self,
        *,
        input_dimension: int,
        bucket_count: int,
        config: RouterConfig,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or bucket_count <= 0:
            raise ValueError("router dimensions must be positive")
        dimensions = [input_dimension, *config.hidden_dimensions, bucket_count]
        activation_type: type[nn.Module] = (
            nn.ReLU if config.activation == "relu" else nn.SiLU
        )
        layers: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(dimensions, dimensions[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(dimensions) - 2:
                layers.append(activation_type())
        self.network = nn.Sequential(*layers)
        self.input_dimension = input_dimension
        self.bucket_count = bucket_count
        self.router_config = config

    @property
    def identity(self) -> str:
        return "raw-query-router-v1"

    def forward(self, queries: Tensor, *, out: Tensor | None = None,
                hidden_out: tuple[Tensor, ...] | None = None) -> Tensor:
        if queries.ndim != 2 or queries.shape[1] != self.input_dimension:
            raise ValueError("query matrix does not match the router input dimension")
        if out is None:
            if hidden_out is not None:
                raise ValueError("hidden output reuse requires a logits output buffer")
            return self.network(queries)
        if self.training or torch.is_grad_enabled():
            raise ValueError("Router output reuse requires eval and disabled gradients")
        if out.shape != (len(queries), self.bucket_count):
            raise ValueError("Router output buffer must match the batch and bucket count")
        output_layer = self.network[-1]
        hidden_index = 0
        for layer in self.network:
            if layer is output_layer:
                return torch.ops.aten.linear.out(
                    queries, layer.weight, layer.bias, out=out,
                )
            if hidden_out is None:
                queries = layer(queries)
            elif isinstance(layer, nn.Linear):
                if hidden_index >= len(hidden_out):
                    raise ValueError("missing hidden output buffer")
                destination = hidden_out[hidden_index][:len(queries)]
                if destination.shape != (len(queries), layer.out_features):
                    raise ValueError("hidden output buffer does not match the layer")
                queries = torch.ops.aten.linear.out(queries, layer.weight, layer.bias, out=destination)
                hidden_index += 1
            elif isinstance(layer, nn.SiLU):
                queries = torch.nn.functional.silu(queries, inplace=True)
            elif isinstance(layer, nn.ReLU):
                queries = torch.relu_(queries)
            else:
                raise ValueError("hidden output reuse supports only Linear, SiLU and ReLU layers")
