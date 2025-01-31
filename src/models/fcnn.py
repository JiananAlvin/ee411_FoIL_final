from __future__ import annotations
import torch.nn as nn
from math import sqrt


class FCNN(nn.Module):
    def __init__(
        self,
        in_nodes: int,
        hidden_nodes: int,
        out_nodes: int,
        final_activation: str,
        weight_reuse: str,
        weight_initialization: str,
        interpolation_threshold: int,
        dropout: float,
        smaller_model: FCNN,
    ):

        super().__init__()

        self.hidden_layer = nn.Linear(in_nodes, hidden_nodes)
        self.dropout_layer = nn.Dropout(dropout)
        self.relu_act = nn.ReLU()
        self.out_layer = nn.Linear(hidden_nodes, out_nodes)

        if final_activation == "none":
            self.post_activation = None
        elif final_activation == "softmax":
            self.post_activation = nn.Softmax(dim=1)
        else:
            exit(f"error: final_activation '{final_activation}' not recognized")

        self.n_parameters = sum(p.numel() for p in self.parameters())
        self._init_weights(
            weight_reuse=weight_reuse,
            weight_initialization=weight_initialization,
            interp_threshold=interpolation_threshold,
            smaller_model=smaller_model,
        )

    def _init_weights(
        self,
        weight_reuse: str,
        weight_initialization: str,
        interp_threshold: int,
        smaller_model: FCNN,
    ):
        cond_no_reuse = (
            smaller_model is None
            or weight_reuse == "none"
            or (
                weight_reuse == "under-parametrized"
                and self.n_parameters > interp_threshold
            )
        )
        if cond_no_reuse:
            if weight_initialization == "xavier_uniform":
                init_ = nn.init.xavier_uniform_
            else:
                exit(
                    f"error: weight_initialization '{weight_initialization}' not recognized"
                )

            init_(self.hidden_layer.weight, gain=1.0)
            nn.init.constant_(self.hidden_layer.bias, 0.01)
            init_(self.out_layer.weight, gain=1.0)
            nn.init.constant_(self.out_layer.bias, 0.01)
        else:
            nn.init.normal_(self.hidden_layer.weight, mean=0.0, std=sqrt(0.01))
            nn.init.normal_(self.hidden_layer.bias, mean=0.0, std=sqrt(0.01))
            nn.init.normal_(self.out_layer.weight, mean=0.0, std=sqrt(0.01))
            nn.init.normal_(self.out_layer.bias, mean=0.0, std=sqrt(0.01))

            hidden_sm = smaller_model.hidden_layer.out_features
            self.hidden_layer.weight.data[:hidden_sm, :] = (
                smaller_model.hidden_layer.weight.data
            )
            self.hidden_layer.bias.data[:hidden_sm] = (
                smaller_model.hidden_layer.bias.data
            )
            self.out_layer.weight.data[:, :hidden_sm] = (
                smaller_model.out_layer.weight.data
            )
            self.out_layer.bias.data = smaller_model.out_layer.bias.data

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.dropout_layer(x)
        x = self.relu_act(x)
        x = self.out_layer(x)
        if self.post_activation is not None:
            x = self.post_activation(x)
        return x
