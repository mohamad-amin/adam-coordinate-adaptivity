import torch


# We could do this in parallel with jax
def construct_orthogonal_rotators(w):
    gaussian_rotators = [gaussian_orthogonal_matrix(d, d, w.device) for d in w.shape]
    return gaussian_rotators


def gaussian_orthogonal_matrix(n, d, device) -> torch.Tensor:
    size = max(n, d)
    torch.cuda.empty_cache()
    gaussian_matrix = torch.randn(size, size, device=device)
    gaussian_matrix, r = torch.linalg.qr(gaussian_matrix)
    del r
    gaussian_matrix = gaussian_matrix[:n, :d]
    torch.cuda.empty_cache()
    return gaussian_matrix
