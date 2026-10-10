import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("decider_reference", Path(__file__).with_name("verify_reference.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceProvenance(unittest.TestCase):
    def test_source_hashes_reject_modified_and_missing_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_bytes(b"abc")
            expected = {"model.py": hashlib.sha256(b"abc").hexdigest()}
            self.assertEqual(module.verify_sources(root, expected), expected)
            source.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "reference checksum"):
                module.verify_sources(root, expected)
            source.unlink()
            with self.assertRaises(FileNotFoundError):
                module.verify_sources(root, expected)


class ResponseFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.responses = [
            {"model": "decider-2b-v11", "usage": {"input_tokens": 1, "output_tokens": 0},
             "answers": fixture["expected_answers"]}
            for fixture in json.loads(Path(__file__).with_name("data").joinpath("responses.json").read_text())
        ]

    def answer(self, kind):
        response = copy.deepcopy(next(r for r in self.responses if any(a["type"] == kind for a in r["answers"].values())))
        key = next(k for k, a in response["answers"].items() if a["type"] == kind)
        return response, key

    def test_all_committed_rounded_assemblies_including_zero_mass_are_valid(self):
        for response in self.responses:
            with self.subTest(answers=response["answers"]):
                module.compare(response, copy.deepcopy(response))

    def test_every_numeric_field_rejects_nonfinite_bool_and_wrong_types(self):
        for kind, fields in (("choice", ("confidence", "certainty", "x_p_max", "probabilities")),
                             ("score", ("confidence", "certainty", "x_p_max", "score", "level_fit", "fit_mass", "probabilities")),
                             ("noul", ("noul",))):
            reference, key = self.answer(kind)
            for field in fields:
                for bad in (float("nan"), float("inf"), float("-inf"), True, "0.5", None, []):
                    for corrupt_reference in (False, True):
                        got = copy.deepcopy(reference)
                        if isinstance(got["answers"][key][field], dict):
                            first = next(iter(got["answers"][key][field]))
                            got["answers"][key][field][first] = bad
                        else:
                            got["answers"][key][field] = bad
                        with self.subTest(kind=kind, field=field, bad=bad, reference=corrupt_reference):
                            with self.assertRaises(AssertionError):
                                module.compare(got, reference) if corrupt_reference else module.compare(reference, got)

    def test_auxiliary_fields_must_follow_the_probability_formulas(self):
        for kind in ("choice", "score"):
            reference, key = self.answer(kind)
            fields = ("confidence", "certainty", "x_p_max") + (("fit_mass",) if kind == "score" else ())
            for field in fields:
                got = copy.deepcopy(reference)
                got["answers"][key][field] = 0.0 if reference["answers"][key][field] > 0.5 else 1.0
                with self.subTest(kind=kind, field=field), self.assertRaises(AssertionError):
                    module.compare(reference, got)
                with self.subTest(kind=kind, field=field, same_reference=True), self.assertRaises(AssertionError):
                    module.compare(got, got)

    def test_reported_numeric_precision_matches_pinned_rounding(self):
        for kind, field in (("choice", "confidence"), ("choice", "probabilities"),
                            ("score", "score"), ("score", "level_fit"), ("noul", "noul")):
            response, key = self.answer(kind)
            value = response["answers"][key][field]
            if isinstance(value, dict):
                first = next(iter(value))
                value[first] += 0.000001
            else:
                response["answers"][key][field] += 0.000001
            with self.subTest(kind=kind, field=field), self.assertRaises(AssertionError):
                module.compare(response, response)

    def assembled(self, kind, probabilities, fits=None):
        # Independent oracle evaluates the pinned formulas on unrounded values,
        # then rounds the wire fields. It does not use verifier interval helpers.
        p = probabilities
        n = len(p)
        modal = max(range(n), key=p.__getitem__)
        confidence = (n * p[modal] - 1) / (n - 1)
        if kind == "score":
            uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
            confidence = 1 - sum(x * abs(i - modal) for i, x in enumerate(p)) / uniform
        answer = {"type": kind, "confidence": round(max(0, min(1, confidence)), 4),
                  "certainty": round(max(0, 1 + sum(x * math.log(x) for x in p if x > 0) / math.log(n)), 4),
                  "x_p_max": round(p[modal], 4), "probabilities": {str(i): round(x, 4) for i, x in enumerate(p)}}
        if kind == "choice":
            answer["choice"] = str(modal)
        else:
            answer.update(score=round(sum(i * x for i, x in enumerate(p)), 2),
                          legend={str(i): f"level {i}" for i in range(n)},
                          level_fit={str(i): round(x, 4) for i, x in enumerate(fits)}, fit_mass=round(sum(fits), 4))
        return {"model": "decider-2b-v11", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": {"q": answer}}

    def test_rounding_intervals_accept_entropy_near_zero_and_score_modal_ambiguity(self):
        for response in (self.assembled("choice", [0.000049, 0.999951]),
                         self.assembled("choice", [1 / 255] * 255),
                         self.assembled("score", [0.350001, 0.349999, 0.3], [0.350001, 0.349999, 0.3]),
                         self.assembled("score", [0.349999, 0.350001, 0.3], [0.349999, 0.350001, 0.3]),
                         self.assembled("score", [0.25, 0.75], [1e-8, 3e-8])):
            with self.subTest(answer=response["answers"]):
                module.compare(response, response)

    def test_probability_and_score_fidelity_gates_still_reject_valid_formulas(self):
        reference = self.assembled("choice", [0.5, 0.5])
        with self.assertRaises(AssertionError):
            module.compare(reference, self.assembled("choice", [0.521, 0.479]))
        reference = self.assembled("score", [0.1] * 10, [0.1] * 10)
        probabilities = [0.1] * 10
        probabilities[0] -= 0.015
        probabilities[9] += 0.015
        with self.assertRaisesRegex(AssertionError, "score"):
            module.compare(reference, self.assembled("score", probabilities, probabilities))

    def test_reference_decision_margin_gate_keeps_low_margin_disagreements_visible(self):
        reference = self.assembled("choice", [0.5099, 0.4901])
        result = module.compare(reference, self.assembled("choice", [0.4901, 0.5099]))
        self.assertEqual(result["low_margin_disagreements"], ["q"])

    def test_extra_missing_or_mismatched_keys_are_rejected(self):
        reference, key = self.answer("score")
        for field in ("level_fit", "legend", "probabilities"):
            for extra in (False, True):
                got = copy.deepcopy(reference)
                if extra:
                    got["answers"][key][field]["extra"] = 0
                else:
                    del got["answers"][key][field][next(iter(got["answers"][key][field]))]
                with self.subTest(field=field, extra=extra), self.assertRaises(AssertionError):
                    module.compare(reference, got)
        for container in ((), ("usage",), ("answers", key)):
            for extra in (False, True):
                got = copy.deepcopy(reference)
                target = got
                for part in container:
                    target = target[part]
                if extra:
                    target["extra"] = 0
                else:
                    del target[next(iter(target))]
                with self.subTest(container=container, extra=extra), self.assertRaises(AssertionError):
                    module.compare(reference, got)

    def test_numeric_ranges_and_usage_integer_types_are_enforced(self):
        for kind, field, bad in (("noul", "noul", -0.1), ("noul", "noul", 1.1),
                                 ("choice", "confidence", -0.1), ("choice", "certainty", 1.1),
                                 ("score", "score", -0.01), ("score", "fit_mass", 11.0)):
            reference, key = self.answer(kind)
            got = copy.deepcopy(reference)
            got["answers"][key][field] = bad
            with self.subTest(kind=kind, field=field, bad=bad), self.assertRaises(AssertionError):
                module.compare(reference, got)
        reference, _ = self.answer("choice")
        for bad in (True, -1, 1.0, "1"):
            got = copy.deepcopy(reference)
            got["usage"]["input_tokens"] = bad
            with self.subTest(usage=bad), self.assertRaises(AssertionError):
                module.compare(got, got)

    def test_original_gates_are_unchanged_and_protocol_declares_new_checks(self):
        self.assertEqual(module.GATES, {"max_probability_drift": 0.02, "max_score_drift": 0.1, "discrete_margin": 0.05})
        self.assertEqual(module.VALIDATION_PROTOCOL["version"], 2)
        self.assertEqual(module.VALIDATION_PROTOCOL["fit_mass_reference_bound"], "number_of_levels * max_probability_drift + 2 * (number_of_levels + 1) * probability_rounding_half_unit")


if __name__ == "__main__":
    unittest.main()
