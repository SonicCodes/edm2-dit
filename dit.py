import math
from typing import Any, Tuple, Optional, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from common import TimestepEmbedder, LabelEmbedder
import numpy as np
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def mp_add(a, b, t=0.3):
    """Magnitude-preserving addition with blend factor t"""
    return ((1-t)*a + t*b) / jnp.sqrt((1-t)**2 + t**2)

def mp_silu(x):
    """Magnitude-preserving SiLU activation"""
    return nn.silu(x) / 0.596

def normalize_weights(w: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """Force normalize weights to unit norm."""
    dim = tuple(range(1, w.ndim))
    norm = jnp.linalg.norm(w, axis=dim, keepdims=True)
    alpha = jnp.sqrt(w[0].size) # fan_in
    return w / (norm * alpha + eps)

def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0) # (1, H*W, D)

DEFAULT_DTYPE = jnp.float32

def pixel_norm(x, eps=1e-4):
    """PixelNorm from EDM2 paper"""
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)

class NormalizedDense(nn.Module):
    features: int
    use_bias: bool = False
    dtype: Any = DEFAULT_DTYPE
    kernel_init: Callable = nn.initializers.normal(stddev=0.02)
    zero_init: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel_init = nn.initializers.zeros if self.zero_init else self.kernel_init
        kernel = self.param('kernel',
                          kernel_init,
                          (inputs.shape[-1], self.features))
        kernel = normalize_weights(kernel)
        kernel = kernel / jnp.sqrt(inputs.shape[-1])
        y = jnp.dot(inputs, kernel)
        return y

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Any = jnp.bfloat16
    out_dim: Optional[int] = None
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = NormalizedDense(self.mlp_dim, dtype=self.dtype)(inputs)
        x=pixel_norm(x)
        x = mp_silu(x)
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(x)
        output = NormalizedDense(actual_out_dim, dtype=self.dtype)(x)
        output = pixel_norm(output)
        if self.dropout_rate > 0.0:
            output = nn.Dropout(rate=self.dropout_rate)(output)
        return output

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

def clb(title, value):
    titles = ["X", "X fin"]
    print(f"{titles[title]}: {value.shape} mean:{np.array(value).mean():.3f} std:{np.array(value).std():.3f} sum:{np.array(value).sum():.3f}")
class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # print("X: ", x.shape, f" mean:{np.array(x).mean():.3f} std:{np.array(x).std():.3f} sum:{np.array(x).sum():.3f}")
        # jax.pure_callback(clb, None, 0, x)
        # Track residual scale with learned gain
        attn_gain = self.param('attn_gain', nn.initializers.zeros, ())
        mlp_gain = self.param('mlp_gain', nn.initializers.zeros, ())

        # Conditioning
        c = mp_silu(c)
        c = NormalizedDense(2 * self.hidden_size, zero_init=True)(c)
        gain, shift = jnp.split(c, 2, axis=-1)
        # print("Cond: ", c.shape, f" mean:{c.mean():.3f} std:{c.std():.3f} sum:{c.sum():.3f}")

        # Attention block
        x_norm = pixel_norm(x)
        # print("X_norm: ", x_norm.shape, f" mean:{x_norm.mean():.3f} std:{x_norm.std():.3f} sum:{x_norm.sum():.3f}")

        x_cond = x_norm * (1 + gain[:, None]) + shift[:, None]
        # print("X_cond: ", x_cond.shape, f" mean:{x_cond.mean():.3f} std:{x_cond.std():.3f} sum:{x_cond.sum():.3f}")

        # QKV with proper scaling
        qkv = NormalizedDense(self.hidden_size * 3)(x_cond)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = pixel_norm(q)
        k = pixel_norm(k)
        v = pixel_norm(v)  # Important - v needs normalization too
        # print("QKV: ", qkv.shape, f" mean:{qkv.mean():.3f} std:{qkv.std():.3f} sum:{qkv.sum():.3f}")

        # Split heads and normalize Q,K
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Normalize q,k for cosine attention
        q = q / jnp.sqrt(jnp.sum(q**2, axis=-1, keepdims=True) + 1e-6)
        k = k / jnp.sqrt(jnp.sum(k**2, axis=-1, keepdims=True) + 1e-6)
        # print("Q normalized: ", q.shape, f" mean:{q.mean():.3f} std:{q.std():.3f} sum:{q.sum():.3f}")
        # print("K normalized: ", k.shape, f" mean:{k.mean():.3f} std:{k.std():.3f} sum:{k.sum():.3f}")

        # Scaled cosine attention
        attn = jnp.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(q.shape[-1])
        attn = nn.softmax(attn)
        # print("Attention: ", attn.shape, f" mean:{attn.mean():.3f} std:{attn.std():.3f} sum:{attn.sum():.3f}")

        # Combine values
        x_attn = jnp.einsum('bhnm,bhmd->bhnd', attn, v)
        x_attn = rearrange(x_attn, 'b h n d -> b n (h d)')
        x_attn = NormalizedDense(self.hidden_size)(x_attn)
        x_attn = pixel_norm(x_attn)
        # print("X_attn pre-gain: ", x_attn.shape, f" mean:{x_attn.mean():.3f} std:{x_attn.std():.3f} sum:{x_attn.sum():.3f}")

        # Apply gain and combine with residual
        x_attn = x_attn * jnp.exp(attn_gain)
        # print("X_attn post-gain: ", x_attn.shape, f" mean:{x_attn.mean():.3f} std:{x_attn.std():.3f} sum:{x_attn.sum():.3f}")
        # x = pixel_norm(x)  # Normalize main path to unit variance
        # x_attn = pixel_norm(x_attn)
        x = mp_add(x, x_attn, t=0.3)
        # print("X post-attn: ", x.shape, f" mean:{x.mean():.3f} std:{x.std():.3f} sum:{x.sum():.3f}")

        # MLP block
        x_norm = pixel_norm(x)
        x_cond = x_norm * (1 + gain[:, None]) + shift[:, None]
        x_mlp = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_cond)
        # print("X_mlp pre-gain: ", x_mlp.shape, f" mean:{x_mlp.mean():.3f} std:{x_mlp.std():.3f} sum:{x_mlp.sum():.3f}")
        x_mlp = x_mlp * jnp.exp(mlp_gain)
        # print("X_mlp post-gain: ", x_mlp.shape, f" mean:{x_mlp.mean():.3f} std:{x_mlp.std():.3f} sum:{x_mlp.sum():.3f}")
        # x_mlp = pixel_norm(x_mlp)  # Already normalized
        x = mp_add(x, x_mlp, t=0.3)
        # print("X final: ", x.shape, f" mean:{x.mean():.3f} std:{x.std():.3f} sum:{x.sum():.3f}")
        # jax.pure_
        #
        # jax.pure_callback(clb, None, 1, x)
        # print("-" * 80)

        return x

class FinalLayer(nn.Module):
    patch_size: int
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c = mp_silu(c)

        x = pixel_norm(x)

        # Add learnable gain parameter
        gain = self.param('gain', nn.initializers.zeros, ())
        x = x * jnp.exp(gain)

        x = NormalizedDense(self.patch_size * self.patch_size * self.out_channels, zero_init=True)(x)
        return x

class DiT(nn.Module):
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int
    learn_sigma: bool = False

    @nn.compact
    def __call__(self, x, t, y, train=False, force_drop_ids=None):
        x = x.transpose((0, 2, 3, 1))
        batch_size, input_size = x.shape[0], x.shape[1]
        in_channels = x.shape[-1]
        out_channels = in_channels if not self.learn_sigma else in_channels * 2

        # Patch embedding
        num_patches = input_size // self.patch_size
        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)',
                     p1=self.patch_size, p2=self.patch_size)
        x = NormalizedDense(self.hidden_size)(x)
        x = pixel_norm(x)

        # Add position embeddings
        pos_embed = self.param("pos_embed", get_2d_sincos_pos_embed, self.hidden_size, x.shape[1])
        pos_embed = jax.lax.stop_gradient(pos_embed)
        x = x + pos_embed

        # Get conditioning
        t = TimestepEmbedder(self.hidden_size)(t)
        y = LabelEmbedder(self.class_dropout_prob, self.num_classes, self.hidden_size)(y, train, force_drop_ids)
        c = mp_add(t, y)

        # Apply transformer blocks
        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, c)

        # Final layer
        x = FinalLayer(self.patch_size, out_channels, self.hidden_size)(x, c)

        # Reshape output
        x = jnp.reshape(x, (batch_size, num_patches, num_patches,
                         self.patch_size, self.patch_size, out_channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C')
        x = x.transpose((0, 3, 1, 2))

        return x


def DiT1M(patch_size, num_classes, class_dropout_prob):
    return DiT(patch_size, 96, 4, 4, 4.0, class_dropout_prob, num_classes)

def DiT10M(patch_size, num_classes, class_dropout_prob):
    return DiT(patch_size, 256, 9, 4, 4.0, class_dropout_prob, num_classes)

def DiT50M(patch_size, num_classes, class_dropout_prob):
    return DiT(patch_size, 480, 12, 8, 4.0, class_dropout_prob, num_classes)

def DiT100M(patch_size, num_classes, class_dropout_prob):
    return DiT(patch_size, 512, 16, 16, 4.0, class_dropout_prob, num_classes)

def DiTXL(patch_size, num_classes, class_dropout_prob):
    return DiT(patch_size, 1152, 28, 16, 4.0, class_dropout_prob, num_classes)

# tests to make sure :3
def test_DiT(size=10, scan_blocks=False):
    # initialize model
    fake_input = jnp.ones((1, 4, 32, 32), jnp.float32)
    fake_t = jnp.ones((1,), jnp.int32)
    fake_y = jnp.ones((1,), jnp.int32)
    if size == 10:
        model = DiT10M(2, 10, 0.1)
    elif size == 50:
        model = DiT50M(2, 10, 0.1)
    elif size == 100:
        model = DiT100M(2, 10, 0.1)
    elif size == 500:
        model = DiTXL(2, 10, 0.1)

    # tabulated_output = nn.tabulate(model, jax.random.key(0), compute_flops=True, depth=1)
    # print(tabulated_output(x=fake_input, y=fake_y, t=fake_t))
    #
    print(" Fake input shape: ", f"{fake_input.shape}, mean: {fake_input.mean()}, std: {fake_input.std()}")

    params = model.init(jax.random.PRNGKey(0), fake_input, fake_t, fake_y, train=True)
    print("Parameter shapes:")
    # test forward pass
    y = model.apply(params, fake_input, fake_t, fake_y, train=True, rngs={'label_dropout': jax.random.PRNGKey(0)})
    print("Model output shape: ", f"{y.shape}, mean: {y.mean()}, std: {y.std()}")
    # how many parameters?
    n_params = sum([p.size for p in jax.tree.leaves(params)])
    print(y.shape, n_params)
    assert y.shape == (1, 4, 32, 32)

if __name__ == "__main__":
    test_DiT(10)
    # test_DiT(50)
    # test_DiT(100)
    # test_DiT(500)
    # test_edm2_properties()
    print("All tests passed!")
