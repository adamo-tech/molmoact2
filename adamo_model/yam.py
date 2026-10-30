from adamo_model import trainer
from adamo_model.policy import load


class Trainer(trainer.Trainer):
    base_model = "allenai/MolmoAct2-BimanualYAM"
