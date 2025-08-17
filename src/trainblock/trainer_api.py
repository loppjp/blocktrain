from typing import Any

class ITrainer:

    def train(self):
        pass

    def eval(self):
        pass

    def save_artifact(self, key):
        pass

    def load_artifact(self, key) -> Any:
        pass