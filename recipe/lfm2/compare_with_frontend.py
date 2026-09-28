#!/usr/bin/env python3
"""Check that a running LFM2 frontend returns exactly what its worker returns.

Sends health, valid choice, stability, unicode and malformed/invalid requests to
the worker directly and through the frontend, then compares status, Content-Type
and raw body bytes. Only choice questions are covered (the worker supports no
other type). Every case also asserts its expected status, so a matching wrong
status still fails. Standard library only; the script never starts or stops the
services and only talks to loopback URLs.
"""

import argparse
import base64
import ipaddress
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

MODEL = "lfm2.5-350m"
STATE = "I was charged twice for my order. Please refund the duplicate today."
TIMEOUT = 90

TARGET = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
}
OTHER = {
    "type": "choice",
    "instructions": "How urgent is the request?",
    "criteria": {"low": "Not urgent", "high": "Needs attention immediately"},
}
UNICODE = {
    "type": "choice",
    "instructions": "Pick the matching label.",
    "criteria": {
        "r\u00e9clamation": "Remboursement",
        "\u65e5\u672c\u8a9e": "\u8fd4\u91d1",
        "\U0001F600 smile": "ok",
        "quote\"backslash\\key": "value",
    },
}
SINGLE = {
    "type": "choice",
    "instructions": "Pick the only option.",
    "criteria": {"only": "The only option"},
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def payload(questions, model=MODEL, state=STATE, ensure_ascii=False):
    envelope = {"model": model, "state": state, "questions": questions}
    return json.dumps(envelope, ensure_ascii=ensure_ascii).encode("utf-8")


def fetch(base, path, body=None):
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers)
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as error:
        response = error
    except (urllib.error.URLError, OSError) as error:
        return {"status": None, "content_type": None, "body": b"", "error": str(error)}
    with response:
        return {
            "status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "body": response.read(),
            "error": None,
        }


def response_evidence(response):
    return {
        "status": response["status"],
        "content_type": response["content_type"],
        "body_b64": base64.b64encode(response["body"]).decode("ascii"),
        "body_text": response["body"].decode("utf-8", errors="replace"),
        "error": response["error"],
    }


def request_evidence(body):
    if body is None:
        return None
    return {
        "body_b64": base64.b64encode(body).decode("ascii"),
        "body_text": body.decode("utf-8", errors="replace"),
    }


def expected_keys(questions):
    return {qid: list(question["criteria"].keys()) for qid, question in questions.items()}


def validate_choice_answers(body, expected):
    """Confirm every choice answer carries a full probability map and a valid pick."""
    details = {}
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return {"error": f"body is not JSON: {error}"}, False
    answers = data.get("answers") if isinstance(data, dict) else None
    ok = isinstance(answers, dict) and set(answers) == set(expected)
    details["qids_match"] = ok
    per = {}
    for qid, keys in expected.items():
        answer = answers.get(qid) if isinstance(answers, dict) else None
        item = {}
        if not isinstance(answer, dict):
            item["ok"] = False
            item["error"] = "answer is not an object"
            per[qid] = item
            ok = False
            continue
        probabilities = answer.get("probabilities")
        numeric = isinstance(probabilities, dict) and probabilities and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in probabilities.values())
        type_ok = answer.get("type") == "choice"
        keys_ok = isinstance(probabilities, dict) and set(probabilities) == set(keys)
        chosen = answer.get("choice")
        argmax = None
        if numeric:
            best_value = None
            for key in keys:
                if key not in probabilities:
                    continue
                value = probabilities[key]
                if best_value is None or value > best_value:
                    best_value = value
                    argmax = key
        argmax_ok = argmax is not None and chosen == argmax
        total = sum(probabilities.values()) if numeric else None
        sum_ok = total is not None and abs(total - 1.0) <= 1e-6
        confidence_value = answer.get("confidence")
        confidence_ok = (isinstance(confidence_value, (int, float))
                         and not isinstance(confidence_value, bool)
                         and 0.0 <= confidence_value <= 1.0)
        item.update({
            "type": answer.get("type"),
            "choice": chosen,
            "argmax": argmax,
            "keys": sorted(probabilities) if isinstance(probabilities, dict) else None,
            "probability_sum": total,
            "confidence": confidence_value,
            "type_ok": type_ok,
            "keys_ok": keys_ok,
            "argmax_ok": argmax_ok,
            "sum_ok": sum_ok,
            "confidence_ok": confidence_ok,
        })
        item["ok"] = type_ok and keys_ok and argmax_ok and sum_ok and confidence_ok
        per[qid] = item
        ok = ok and item["ok"]
    details["answers"] = per
    return details, ok


def run_case(worker_url, frontend_url, name, path, body, expected, questions=None):
    direct = fetch(worker_url, path, body)
    proxied = fetch(frontend_url, path, body)
    direct_status, frontend_status = direct["status"], proxied["status"]
    checks = {
        "direct_status": direct_status,
        "frontend_status": frontend_status,
        "expected_status": expected,
        "status_expected": direct_status == expected and frontend_status == expected,
        "status_equal": direct_status == frontend_status,
        "content_type_equal": direct["content_type"] == proxied["content_type"],
        "content_type": proxied["content_type"],
        "body_equal": direct["body"] == proxied["body"],
    }
    ok = all((checks["status_expected"], checks["status_equal"],
              checks["content_type_equal"], checks["body_equal"]))
    if questions is not None and frontend_status == 200:
        details, semantic_ok = validate_choice_answers(proxied["body"], expected_keys(questions))
        checks["semantics"] = details
        ok = ok and semantic_ok
    evidence = {
        "name": name,
        "path": path,
        "expected_status": expected,
        "request": request_evidence(body),
        "direct": response_evidence(direct),
        "frontend": response_evidence(proxied),
        "checks": checks,
        "pass": ok,
    }
    return evidence, proxied


def stability_case(worker_url, frontend_url):
    variants = [
        ("base", {"target": TARGET}, "target"),
        ("rename", {"renamed_target": TARGET}, "renamed_target"),
        ("reorder", {"intro": OTHER, "target": TARGET}, "target"),
    ]
    sub = []
    answers = {}
    ok = True
    for label, questions, qid in variants:
        evidence, proxied = run_case(worker_url, frontend_url, f"stability/{label}",
                                     "/v1/systemone", payload(questions), 200, questions)
        sub.append(evidence)
        ok = ok and evidence["pass"]
        if proxied["status"] == 200:
            try:
                answers[label] = json.loads(proxied["body"])["answers"][qid]
            except (KeyError, TypeError, ValueError):
                ok = False
    checks = {"variants_run": len(answers)}
    if len(answers) == 3:
        choices = {label: answer.get("choice") for label, answer in answers.items()}
        checks["choices"] = choices
        checks["choice_stable"] = len(set(choices.values())) == 1
        base = answers["base"].get("probabilities") or {}
        keys = set(base)
        for answer in answers.values():
            keys |= set(answer.get("probabilities") or {})
        max_diff = 0.0
        for label in ("rename", "reorder"):
            probabilities = answers[label].get("probabilities") or {}
            for key in keys:
                max_diff = max(max_diff, abs(base.get(key, 0.0) - probabilities.get(key, 0.0)))
        checks["max_probability_diff"] = max_diff
        checks["probability_stable"] = bool(keys) and max_diff <= 1e-6
        ok = ok and checks["choice_stable"] and checks["probability_stable"]
    else:
        checks["choice_stable"] = False
        checks["probability_stable"] = False
    return {
        "name": "rename_reorder_stability",
        "path": "/v1/systemone",
        "expected_status": 200,
        "variants": sub,
        "checks": checks,
        "pass": ok,
    }


def ensure_loopback(url, label):
    host = urllib.parse.urlsplit(url).hostname
    if host is None:
        raise ValueError(f"{label} URL has no host: {url!r}")
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"{label} URL must target loopback, got {host!r}")
    if not (address.is_loopback or address.is_unspecified):
        raise ValueError(f"{label} URL must target loopback, got {host!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:8080")
    parser.add_argument("--output", required=True, help="path for the JSON report")
    args = parser.parse_args()

    try:
        ensure_loopback(args.worker, "worker")
        ensure_loopback(args.frontend, "frontend")
    except ValueError as error:
        parser.error(str(error))

    multiple = {"billing_q": TARGET, "urgency_q": OTHER,
                "refund_q": {"type": "choice", "instructions": "Refund requested?",
                             "criteria": {"yes": "A refund is requested", "no": "No refund"}}}
    invalid_later = {"first": TARGET,
                     "second": {"type": "noul", "instructions": "Ask for a refund?"}}

    cases = [
        ("health", "/health", None, 200, None),
        ("singlechoice", "/v1/systemone", payload({"target": TARGET}), 200, {"target": TARGET}),
        ("multiplechoice", "/v1/systemone", payload(multiple), 200, multiple),
        ("unicode_escaped", "/v1/systemone", payload({"labels": UNICODE}, ensure_ascii=True), 200,
         {"labels": UNICODE}),
        ("singlecandidate", "/v1/systemone", payload({"solo": SINGLE}), 200, {"solo": SINGLE}),
        ("invalidscore422", "/v1/systemone",
         payload({"q": {"type": "score", "instructions": "Rate", "criteria": {"a": "A", "b": "B"}}}),
         422, None),
        ("invalidmodel422", "/v1/systemone",
         payload({"q": TARGET}, model="not-" + MODEL), 422, None),
        ("invalidlaterquestion422", "/v1/systemone", payload(invalid_later), 422, None),
        ("malformedJSON400", "/v1/systemone", b'{"model": "lfm2.5-350m", "state":', 400, None),
    ]

    results = []
    passed = 0
    for name, path, body, expected, questions in cases:
        evidence, _ = run_case(args.worker, args.frontend, name, path, body, expected, questions)
        results.append(evidence)
        passed += evidence["pass"]
        status = "PASS" if evidence["pass"] else "FAIL"
        checks = evidence["checks"]
        print(f"{status} {name}: expected {expected} "
              f"direct {checks['direct_status']} frontend {checks['frontend_status']} "
              f"bytes_equal {checks['body_equal']}")

    stability = stability_case(args.worker, args.frontend)
    results.append(stability)
    passed += stability["pass"]
    status = "PASS" if stability["pass"] else "FAIL"
    print(f"{status} rename_reorder_stability: expected 200 "
          f"choices {stability['checks'].get('choices')} "
          f"max_probability_diff {stability['checks'].get('max_probability_diff')}")

    total = len(results)
    failed = total - passed
    report = {
        "worker": args.worker,
        "frontend": args.frontend,
        "model": MODEL,
        "timeout_seconds": TIMEOUT,
        "cases": results,
        "passed": passed,
        "failed": failed,
        "pass": failed == 0,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)

    print(f"{passed}/{total} cases passed; report written to {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
