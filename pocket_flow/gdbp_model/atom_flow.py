from typing import override

import torch
from torch import Tensor, nn

from pocket_flow.gdbp_model.layers import GDBLinear, GDBPerceptronVN, ST_GDBP_Exp
from pocket_flow.gdbp_model.types import BottleneckSpec


class AtomFlow(nn.Module):
    """Conditional normalising flow for atom-type features.

    This module parameterises an elementwise affine bijection between a
    dequantised atom-type representation x ∈ R^K (typically a one-hot vector
    plus uniform noise) and a base latent variable z ∈ R^K (typically standard
    Gaussian), conditioned on encoder features at the focal atoms.

    For each focal atom i, let h_i be the context features produced by `self.net`.
    Each flow layer predicts elementwise log-scale and translation parameters
    (log s_i, t_i) from h_i only, and applies a diagonal affine transform:

        z = (x + t(h)) ⊙ exp(log s(h))

    With the identification t(h) = -μ(h) and exp(log s(h)) = 1/σ(h), this matches
    the common affine flow form z = (x - μ(h)) / σ(h).

    `forward` implements the data→latent direction (x → z) and returns both z and
    the *elementwise* log-Jacobian contributions. To obtain a scalar log|det J|
    per atom, sum the returned log-Jacobian over the last dimension.

    `reverse` implements the latent→data direction (z → x).
    """

    net: nn.Sequential
    flow_layers: nn.ModuleList

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        hidden_dim_sca: int,
        hidden_dim_vec: int,
        num_lig_atom_type: int,
        num_flow_layers: int,
        bottleneck: BottleneckSpec,
        use_conv1d: bool = False,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            GDBPerceptronVN(
                in_sca,
                in_vec,
                hidden_dim_sca,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
            GDBLinear(
                hidden_dim_sca,
                hidden_dim_vec,
                hidden_dim_sca,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            ),
        )

        self.flow_layers = nn.ModuleList()
        for _ in range(num_flow_layers):
            layer = ST_GDBP_Exp(
                hidden_dim_sca,
                hidden_dim_vec,
                num_lig_atom_type,
                hidden_dim_vec,
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            )
            self.flow_layers.append(layer)

    @override
    def forward(
        self,
        x_atom: Tensor,
        compose_features: tuple[Tensor, Tensor],
        focal_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Map dequantised atom-type features to latent space.

        Args:
            x_atom:
                Dequantised atom-type representation in data space,
                shape (N_focal, K) where K = num_lig_atom_type.
            compose_features:
                Tuple (scalar_features, vector_features) from the encoder,
                each indexed by `focal_idx`:
                  - scalar_features: (N_compose, hidden_dim_sca)
                  - vector_features: (N_compose, hidden_dim_vec, 3)
            focal_idx:
                Indices of focal atoms into `compose_features`, shape (N_focal,).

        Returns:
            z_latent, logabsdet_elementwise:
              - z_latent: latent representation, shape (N_focal, K)
              - logabsdet_elementwise: elementwise log|∂z/∂x| contributions,
                shape (N_focal, K). Sum over the last dimension to obtain a
                scalar log|det J| per focal atom.

        Notes:
            If N_focal == 0, this is the identity map and returns zeros for the
            log-Jacobian contributions.
        """
        if focal_idx.size(0) == 0:
            atom_log_jacob = torch.zeros_like(x_atom)
            return x_atom, atom_log_jacob

        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])

        z_atom = x_atom
        atom_log_jacob = torch.zeros_like(z_atom)
        for flow_layer in self.flow_layers:
            log_s, t = flow_layer([sca_focal, vec_focal])
            z_atom = (z_atom + t) * torch.exp(log_s)
            atom_log_jacob += log_s

        return z_atom, atom_log_jacob

    def reverse(
        self,
        atom_latent: Tensor,
        compose_features: tuple[Tensor, Tensor],
        focal_idx: Tensor,
    ) -> Tensor:
        """Map latent samples to dequantised atom-type features.

        This applies the inverse of `forward`, i.e. latent→data:

            x = z ⊙ exp(-log s(h)) - t(h)

        Args:
            atom_latent:
                Latent samples in base space, shape (N_focal, K).
            compose_features:
                Tuple (scalar_features, vector_features) from the encoder:
                  - scalar_features: (N_compose, hidden_dim_sca)
                  - vector_features: (N_compose, hidden_dim_vec, 3)
            focal_idx:
                Indices of focal atoms, shape (N_focal,).

        Returns:
            Dequantised atom-type features in data space, shape (N_focal, K).

        Notes:
            If N_focal == 0, returns `atom_latent` unchanged.
        """
        if focal_idx.size(0) == 0:
            return atom_latent

        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])

        for flow_layer in reversed(self.flow_layers):
            log_s, t = flow_layer([sca_focal, vec_focal])
            inv_scale = torch.exp(-log_s)
            atom_latent = atom_latent * inv_scale - t

        return atom_latent
