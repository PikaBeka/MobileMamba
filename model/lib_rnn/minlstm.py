import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence


def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class minLSTM(Module):
    def __init__(self, dim, expansion_factor=1., proj_out=None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_f_i_gate = Linear(dim, dim_inner * 3, bias=False)
        self.to_out = Linear(
            dim_inner, dim, bias=False) if proj_out else Identity()

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, f_gate, i_gate = self.to_hidden_and_f_i_gate(
            x).chunk(3, dim=-1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            f_gate = f_gate.sigmoid()
            i_gate = i_gate.sigmoid()
            f_gate_prime = f_gate/(f_gate + i_gate)
            i_gate_prime = i_gate/(f_gate + i_gate)
            out = (
                (prev_hidden * f_gate_prime) + (hidden * i_gate_prime) if exists(prev_hidden)
                else (hidden * i_gate_prime)
            )
        else:
            # parallel

            diff = F.softplus(-f_gate) - F.softplus(-i_gate)

            log_f = -F.softplus(diff)
            log_i = -F.softplus(-diff)

            log_h_0 = -F.softplus(log_f)

            log_tilde_h = log_g(hidden)
            log_values = log_i + log_tilde_h

            if exists(prev_hidden):
                log_h_0 = log_g(prev_hidden)
                log_values = torch.cat((log_h_0, log_values), dim=1)
                log_f = F.pad(log_f, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_f, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden


class MinLSTM2D(nn.Module):
    """
    Channel-first image wrapper around minLSTM:
    [B, C, H, W] -> flatten -> [B, L, C] -> minLSTM -> [B, C, H, W]
    """

    def __init__(self, dim, expansion_factor=1., proj_out=None):
        super().__init__()
        self.dim = dim
        self.rnn = minLSTM(
            dim, expansion_factor=expansion_factor, proj_out=proj_out)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()    # [B, H, W, C]
        x = x.view(B, H * W, C)                   # [B, L, C]
        y = self.rnn(x)                           # [B, L, C]
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y
