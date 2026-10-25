import torch
from compass_circuits.data import coordinates, x_only_pairs
from compass_circuits.model import Config, CompassTransformer, hook_site, loss_and_accuracy


def test_coordinates():
    # N N E S E => (2, 1); prefix positions
    tokens = torch.tensor([[0, 0, 1, 2, 1]])
    assert coordinates(tokens)[0].tolist() == [[0, 1], [0, 2], [1, 2], [1, 1], [2, 1]]


def test_forward_backward_and_hook():
    model = CompassTransformer(Config(seq_len=5, d_model=32, n_layers=2, n_heads=4, d_mlp=64))
    tokens = torch.tensor([[0, 0, 1, 2, 1], [3, 1, 3, 0, 2]])
    label = coordinates(tokens)[:, -1, :]
    seen = []
    def observer(_m, _inp, out):
        seen.append(out.shape)
    handle = hook_site(model, 0, "attn_out").register_forward_hook(observer)
    try:
        logits = model(tokens)
    finally:
        handle.remove()
    assert logits[0].shape == (2, 11)
    assert seen == [torch.Size([2, 5, 32])]
    loss, _, _ = loss_and_accuracy(logits, label, seq_len=5)
    loss.backward()
    assert model.blocks[0].qkv.weight.grad is not None


def test_pair_intervention_preserves_y():
    a, b = x_only_pairs(64, 12, torch.device("cpu"))
    difference = coordinates(b)[:, -1] - coordinates(a)[:, -1]
    assert torch.equal(difference, torch.tensor([[2, 0]]).expand(64, 2))


def test_hook_can_replace_activation():
    model = CompassTransformer(Config(seq_len=4, d_model=32, n_layers=1, n_heads=4, d_mlp=64)).eval()
    input_tokens = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        original = model(input_tokens)[0]
        handle = hook_site(model, 0, "attn_out").register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
        try:
            changed = model(input_tokens)[0]
        finally:
            handle.remove()
    assert not torch.allclose(original, changed)
