from blocktrain.trainer_api import ITrainer

def trainer_factory(
    *args,
    training_input_parameters: dict = None,
    **kwargs,
) -> ITrainer:
    pass
