import unittest

import numpy as np
import numpy.testing as np_test
from scipy.stats import norm

from pgmpy.factors.continuous import ConditionalLinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ConditionalLinearGaussianInference
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
    cpds = [
        TabularCPD(
            "device",
            2,
            [[0.5], [0.5]],
            evidence=[],
            evidence_card=[],
            state_names={"device": ["type_a", "type_b"]},
        ),
        ConditionalLinearGaussianCPD(
            variable="temperature",
            discrete_evidence=["device"],
            continuous_evidence=[],
            state_names={"device": ["type_a", "type_b"]},
            parameters={
                ("type_a",): {"beta": [0.0], "std": 1.0},
                ("type_b",): {"beta": [2.0], "std": 1.0},
            },
        ),
        ConditionalLinearGaussianCPD(
            variable="voltage",
            discrete_evidence=["device"],
            continuous_evidence=[],
            state_names={"device": ["type_a", "type_b"]},
            parameters={
                ("type_a",): {"beta": [0.0], "std": 1.0},
                ("type_b",): {"beta": [0.0], "std": 1.0},
            },
        ),
        ConditionalLinearGaussianCPD(
            variable="current",
            discrete_evidence=["device"],
            continuous_evidence=["temperature"],
            state_names={"device": ["type_a", "type_b"]},
            parameters={
                ("type_a",): {"beta": [0.0, 1.0], "std": 0.5},
                ("type_b",): {"beta": [0.0, -1.0], "std": 0.5},
            },
        ),
    ]
    model.add_cpds(*cpds)
    return model


class TestConditionalLinearGaussianInference(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        self.inference = ConditionalLinearGaussianInference(self.model)
        self.evidence = {"temperature": 1.0, "voltage": 0.0}

    def test_posterior_keeps_two_gaussian_components(self):
        posterior = self.inference.query(
            ["device", "current"],
            evidence=self.evidence,
        )

        self.assertEqual(posterior.variables, ["current"])
        self.assertEqual(posterior.discrete_variables, ["device"])
        self.assertEqual(len(posterior.weights), 2)
        np_test.assert_allclose(posterior.weights, [0.5, 0.5])
        np_test.assert_allclose(posterior.means[:, 0], [1.0, -1.0])
        np_test.assert_allclose(
            posterior.covariances,
            np.broadcast_to(np.array([[[0.25]]]), (2, 1, 1)),
        )
        self.assertEqual(
            [state["device"] for state in posterior.discrete_states],
            ["type_a", "type_b"],
        )

    def test_multivariate_result_preserves_variable_order_and_covariance(self):
        posterior = self.inference.query(
            ["device", "current", "temperature"],
            evidence={"voltage": 0.0},
        )

        self.assertEqual(posterior.variables, ["current", "temperature"])
        np_test.assert_allclose(posterior.means, [[0.0, 0.0], [-2.0, 2.0]])
        np_test.assert_allclose(
            posterior.covariances,
            np.array(
                [
                    [[1.25, 1.0], [1.0, 1.0]],
                    [[1.25, -1.0], [-1.0, 1.0]],
                ]
            ),
        )

    def test_discrete_only_query(self):
        posterior = self.inference.query(["device"], evidence=self.evidence)

        np_test.assert_allclose(posterior.values, [0.5, 0.5])
        self.assertEqual(posterior.state_names["device"], ["type_a", "type_b"])

    def test_invalid_evidence(self):
        with self.assertRaisesRegex(ValueError, "Unknown state"):
            self.inference.query(["current"], evidence={"device": "type_c", "voltage": 0.0})
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.inference.query(["current"], evidence={"voltage": np.nan})

    def test_marginal_queries_do_not_collapse_components(self):
        posterior = self.inference.query(
            ["device", "temperature", "current"],
            evidence={"voltage": 0.0},
            joint=False,
        )

        current = posterior["current"]
        self.assertEqual(len(current.weights), 2)
        np_test.assert_allclose(current.weights, [0.5, 0.5])
        np_test.assert_allclose(current.means[:, 0], [0.0, -2.0])
        np_test.assert_allclose(current.variances(), [1.25, 1.25])

    def test_weights_match_enumeration(self):
        likelihood_type_a = norm.pdf(1.0, loc=0.0, scale=1.0) * norm.pdf(0.0, loc=0.0, scale=1.0)
        likelihood_type_b = norm.pdf(1.0, loc=2.0, scale=1.0) * norm.pdf(0.0, loc=0.0, scale=1.0)
        expected_unnormalized = np.array([0.5 * likelihood_type_a, 0.5 * likelihood_type_b])
        expected_weights = expected_unnormalized / expected_unnormalized.sum()

        posterior = self.inference.query(["device"], evidence=self.evidence)

        np_test.assert_allclose(
            posterior.values.ravel(),
            expected_weights,
        )
