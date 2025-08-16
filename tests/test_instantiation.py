import unittest

from blocktrain.factories.experiment import from_file
from blocktrain.test_utilities.path_utilities import get_test_experiment

class TestInstantiateExperiment(unittest.TestCase):

    def test_instantiate(self):

        t = get_test_experiment()
        e = from_file(t)
        print(e)