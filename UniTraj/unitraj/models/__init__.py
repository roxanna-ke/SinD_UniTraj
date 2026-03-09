from importlib import import_module


_MODEL_REGISTRY = {
    'autobot': ('unitraj.models.autobot.autobot', 'AutoBotEgo'),
    'wayformer': ('unitraj.models.wayformer.wayformer', 'Wayformer'),
    'MTR': ('unitraj.models.mtr.MTR', 'MotionTransformer'),
    'MAE': ('unitraj.models.fmae.trainer_mae', 'TrainerMAE'),
    'forecast': ('unitraj.models.fmae.trainer_forecast', 'TrainerForecast'),
    'EMP': ('unitraj.models.emp.trainer_forecast', 'TrainerEMP'),
    'SMART': ('unitraj.models.smart.smart', 'SMART'),
}


def build_model(config):
    module_name, class_name = _MODEL_REGISTRY[config.method.model_name]
    model_cls = getattr(import_module(module_name), class_name)
    model = model_cls(config=config)
    return model
