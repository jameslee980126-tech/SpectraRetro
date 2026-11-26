import os

__all__ = ['load_model']

def load_model(
        model_name: str,
        model_class,
        pretrain_path: str = None
    ):
    if pretrain_path is not None:
        model_path = os.path.join(pretrain_path, model_name)
    else:
        model_path = model_name
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(model_path)
    else:
        model = model_class.from_pretrained(model_path)
    return model
