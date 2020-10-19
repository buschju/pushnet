from typing import Type

from models.pushnet import PushNet

_MODELS = {'PushNet': PushNet,
           }


def get_model_class(model_name: str,
                    ) -> Type:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError('Unknown model: {}'.format(model_name))
