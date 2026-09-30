import unittest

import numpy as np
import numpy.testing as np_test

from pgmpy.factors.continuous import ConditionalLinearGaussianCPD, LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import ConditionalLinearGaussianBayesianNetwork


def build_model():
    model = ConditionalLinearGaussianBayesianNetwork(
        [
            ("device", "temperature"),
            ("device", "voltage"),
            ("device", "current"),
            ("temperature", "current"),
        ],
        discrete_variables=["device"],
    )
    device_cpd = TabularCPD(
        "device",
        2,
        [[0.5], [0.5]],
        evidence=[],
        evidence_card=[],
        state_names={"device": ["type_a", "type_b"]},
    )
    temperature_cpd = ConditionalLinearGaussianCPD(
        variable="temperature",
        discrete_evidence=["device"],
        continuous_evidence=[],
        state_names={"device": ["type_a", "type_b"]},
        parameters={
            ("type_a",): {"beta": [0.0], "std": 1.0},
            ("type_b",): {"beta": [2.0], "std": 1.0},
        },
    )
    voltage_cpd = ConditionalLinearGaussianCPD(
        variable="voltage",
        discrete_evidence=["device"],
        continuous_evidence=[],
        state_names={"device": ["type_a", "type_b"]},
        parameters={
            ("type_a",): {"beta": [0.0], "std": 1.0},
            ("type_b",): {"beta": [0.0], "std": 1.0},
        },
    )
    current_cpd = ConditionalLinearGaussianCPD(
        variable="current",
        discrete_evidence=["device"],
        continuous_evidence=["temperature"],
        state_names={"device": ["type_a", "type_b"]},
        parameters={
            ("type_a",): {"beta": [0.0, 1.0], "std": 0.5},
            ("type_b",): {"beta": [0.0, -1.0], "std": 0.5},
        },
    )
    model.add_cpds(device_cpd, temperature_cpd, voltage_cpd, current_cpd)
    return model


class TestConditionalLinearGaussianBayesianNetwork(unittest.TestCase):
    def test_construction_and_validation(self):
        model = build_model()

        self.assertTrue(model.check_model())
        self.assertEqual(model.discrete_variables, ["device"])
        self.assertEqual(model.continuous_variables, ["temperature", "voltage", "current"])
        self.assertEqual(model.states["device"], ["type_a", "type_b"])

        cpd = model.get_cpds("current")
        beta, std = cpd.get_parameters(("type_a",))
        np_test.assert_array_equal(beta, np.array([0.0, 1.0]))
        self.assertEqual(std, 0.5)

    def test_continuous_parent_of_discrete_node_is_invalid(self):
        model = ConditionalLinearGaussianBayesianNetwork(
            [("temperature", "device")],
            discrete_variables=["device"],
        )
        cpd = TabularCPD(
            "device",
            2,
            [[0.5, 0.5], [0.5, 0.5]],
            evidence=["temperature"],
            evidence_card=[2],
        )
        model.add_cpds(cpd)
        temperature_cpd = ConditionalLinearGaussianCPD(
            variable="temperature",
            discrete_evidence=[],
            continuous_evidence=[],
            parameters={(): {"beta": [0.0], "std": 1.0}},
        )
        model.add_cpds(temperature_cpd)

        with self.assertRaisesRegex(ValueError, "Discrete node.*can only have discrete parents"):
            model.check_model()

    def test_missing_discrete_configuration_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "Missing parameters"):
            ConditionalLinearGaussianCPD(
                variable="temperature",
                discrete_evidence=["device"],
                continuous_evidence=[],
                state_names={"device": ["type_a", "type_b"]},
                parameters={("type_a",): {"beta": [0.0], "std": 1.0}},
            )

    def test_pure_linear_gaussian_cpd_is_accepted_for_continuous_node(self):
        model = ConditionalLinearGaussianBayesianNetwork(
            [("device", "temperature"), ("temperature", "voltage")],
            discrete_variables=["device"],
        )
        device_cpd = TabularCPD(
            "device",
            2,
            [[0.5], [0.5]],
            evidence=[],
            evidence_card=[],
            state_names={"device": ["type_a", "type_b"]},
        )
        temperature_cpd = ConditionalLinearGaussianCPD(
            variable="temperature",
            discrete_evidence=["device"],
            continuous_evidence=[],
            state_names={"device": ["type_a", "type_b"]},
            parameters={
                ("type_a",): {"beta": [0.0], "std": 1.0},
                ("type_b",): {"beta": [1.0], "std": 1.0},
            },
        )
        voltage_cpd = LinearGaussianCPD("voltage", beta=[0.0, 2.0], std=1.0, evidence=["temperature"])
        model.add_cpds(device_cpd, temperature_cpd, voltage_cpd)

        self.assertTrue(model.check_model())
        self.assertEqual(model.get_cpds("voltage").continuous_evidence, ["temperature"])
