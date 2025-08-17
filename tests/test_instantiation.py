import unittest

import yaml
import torch

from blocktrain.factories.loader import load
from blocktrain.factories.experiment_factory import from_file
from blocktrain.test_utilities.path_utilities import get_test_experiment
from blocktrain.component_api import IComponentProvider
from blocktrain.factories.dataloader_factory import dataloader_factory


class TestInstantiateExperiment(unittest.TestCase):

    def test_instantiate(self):

        t = get_test_experiment()
        e = from_file(t)
        self.assertIsNotNone(e)


class TestInstantiateIRDataset(unittest.TestCase):

    def test_instantiate(self):
        t = get_test_experiment()
        spec = yaml.load(t.open(), Loader=yaml.SafeLoader)
        e = load(spec["experiment"]["train_dataset"])
        self.assertIsNotNone(e)


class TestInstantiateModel(unittest.TestCase):

    def test_instantiate(self):
        t = get_test_experiment()
        spec = yaml.load(t.open(), Loader=yaml.SafeLoader)
        m = load(spec["experiment"]["model"])
        self.assertIsNotNone(m)

class TestModel(torch.nn.Module):
    def __init__(self): 
        super().__init__()
        self.w1 = torch.nn.Linear(2, 2)
        self.w2 = torch.nn.Linear(2, 2)
    def forward(self, x):
        return self.w2(self.w1(x))

class TestComponentProvider(IComponentProvider):
    def get_component(self, component_name):
        return component_name

    def get_model(self):
        return TestModel()


class TestInstantiateOptimizer(unittest.TestCase):

    def test_instantiate(self):
        t = get_test_experiment()
        spec = yaml.load(t.open(), Loader=yaml.SafeLoader)
        o = load(
            spec["experiment"]["optimizer"],
            component_provider=TestComponentProvider()
        )
        self.assertIsNotNone(o)


class TestInstantiateTrainer(unittest.TestCase):

    def test_instantiate(self):
        t = get_test_experiment()
        spec = yaml.load(t.open(), Loader=yaml.SafeLoader)
        tr = load(
            spec["experiment"]["trainer"],
            component_provider=TestComponentProvider()
        )
        self.assertIsNotNone(tr)

class TestDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return 0

class TestInstantiateDataloader(unittest.TestCase):

    def test_instantiate(self):
        t = get_test_experiment()
        spec = yaml.load(t.open(), Loader=yaml.SafeLoader)
        dl = dataloader_factory(
            TestDataset(),
            **spec["experiment"]["train_dataloader"],
        )
        self.assertIsNotNone(dl)
