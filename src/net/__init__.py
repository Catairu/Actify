from .models.cnn_lstm import Cnn_Lstm

# from .models.convae_lstm import ConvAeLstm


def get_model(cfg):
    model_type = cfg.get("name", "cnn_lstm")

    if model_type == "cnn_lstm":
        return Cnn_Lstm(cfg)
    elif model_type == "convae_lstm":
        return ConvAeLstm(cfg)
    else:
        raise ValueError(
            f"Modello '{model_type}' non riconosciuto in src/net/__init__.py"
        )
