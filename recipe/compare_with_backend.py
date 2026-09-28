#!/usr/bin/env python3
"""Check that a running omni-jev frontend returns exactly what its worker returns.

Sends the same health and decision requests to the worker directly and through
the frontend, then compares status, Content-Type and body bytes. Covers choice,
score and noul questions separately and together. Standard library only.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

STATE = "I was charged twice for my order. Please refund the duplicate today."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["Not urgent", "Needs attention soon", "Needs attention immediately"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def fetch(base, path, body=None):
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers)
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request, timeout=90)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers.get("Content-Type"), response.read()


def decision(body):
    """The part of a response the frontend must not change: None when body is not a decision.

    ``usage`` is deliberately excluded. A backend that reuses encoder state across requests
    charges only for the calls it paid for, so an identical request can report different
    ``input_tokens`` depending on whether the work was already cached -- CLM does exactly
    this. The answer must still be identical; the envelope is reported, not asserted.
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or "answers" not in parsed:
        return None
    return parsed["answers"], parsed.get("usage")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="model name the worker serves")
    args = parser.parse_args()

    cases = [("health", "/health", None)]
    for name, question in [*QUESTIONS.items(), ("combined", None)]:
        questions = {name: question} if question else QUESTIONS
        payload = {"model": args.model, "state": STATE, "questions": questions}
        cases.append((name, "/v1/systemone", json.dumps(payload).encode()))

    failed = 0
    for name, path, body in cases:
        direct = fetch(args.backend, path, body)
        proxied = fetch(args.frontend, path, body)
        ok = direct == proxied and direct[0] == 200
        note = ""
        if not ok:
            # Compare the decision itself before falling back to the byte comparison.
            left, right = decision(direct[2]), decision(proxied[2])
            if left and right and direct[:2] == proxied[:2] and left[0] == right[0]:
                ok = True
                note = f"  (usage {left[1]} -> {right[1]})"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: status {direct[0]} -> {proxied[0]}{note}")
        if body is not None:
            print(f"     {proxied[2].decode(errors='replace')}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
