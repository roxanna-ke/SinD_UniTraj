from importlib import import_module


_DATASET_REGISTRY = {
    'autobot': ('unitraj.datasets.autobot_dataset', 'AutoBotDataset'),
    'wayformer': ('unitraj.datasets.wayformer_dataset', 'WayformerDataset'),
    'MTR': ('unitraj.datasets.MTR_dataset', 'MTRDataset'),
    'forecast': ('unitraj.datasets.fmae_dataset', 'FMAEDataset'),
    'MAE': ('unitraj.datasets.fmae_dataset', 'FMAEDataset'),
    'EMP': ('unitraj.datasets.EMP_dataset', 'EMPDataset'),
    'SMART': ('unitraj.datasets.SMART_dataset', 'SMARTDataset'),
}


def build_dataset(config, val=False):
    module_name, class_name = _DATASET_REGISTRY[config.method.model_name]
    dataset_cls = getattr(import_module(module_name), class_name)
    dataset = dataset_cls(config=config, is_validation=val)
    return dataset
