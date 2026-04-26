from .deepfm import DeepFM


def create_model(name: str, config: dict):
    """Factory: create model from config."""
    m = config.get("model", {})
    d = config.get("data", {})

    common = dict(
        num_dense=m.get("num_dense", 13),
        num_sparse_fields=m.get("num_sparse_fields", 26),
        hash_bucket_size=d.get("hash_bucket_size", 100_000),
    )

    if name == "deepfm":
        return DeepFM(**common, embed_dim=m.get("embed_dim", 16),
                      hidden_dims=m.get("hidden_dims", [256, 128, 64]),
                      dropout=m.get("dropout", 0.2))
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: deepfm")
