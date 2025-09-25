import numpy as np
import jax.numpy as jnp


def sigmoid_cross_entropy(*, targets, logits):
    """Numerically stable sigmoid cross-entropy.

    Args:
        targets: Array of 0/1 targets (same shape as logits).
        logits: Array of logits (pre-sigmoid activations).

    Returns:
        Elementwise loss with the same shape as inputs.
    """
    # Stable formulation: max(l, 0) - l * y + log(1 + exp(-|l|))
    # Works with both numpy and jax numpy arrays; jnp ops keep JAX compatibility.
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def discriminator_loss(*, real_logits, fake_logits):
    """Discriminator loss for GANs (non-saturating formulation for the generator).

    Expects logits (not probabilities).
    Returns a scalar mean loss.
    """
    real_logits = jnp.asarray(real_logits)
    fake_logits = jnp.asarray(fake_logits)

    real_targets = jnp.ones_like(real_logits)
    fake_targets = jnp.zeros_like(fake_logits)

    real_loss = sigmoid_cross_entropy(targets=real_targets, logits=real_logits)
    fake_loss = sigmoid_cross_entropy(targets=fake_targets, logits=fake_logits)
    loss = jnp.mean(real_loss) + jnp.mean(fake_loss)

    assert loss.shape == (), 'discriminator loss is expected to be a scalar'
    return loss


def generator_loss(discriminator_fake_logits):
    """Generator non-saturating loss: encourage D(G(z)) -> 1.

    Expects discriminator logits on fake samples. Returns a scalar mean loss.
    """
    fake_logits = jnp.asarray(discriminator_fake_logits)
    targets = jnp.ones_like(fake_logits)
    loss = jnp.mean(sigmoid_cross_entropy(targets=targets, logits=fake_logits))

    assert loss.shape == (), 'generator loss is expected to be a scalar'
    return loss


