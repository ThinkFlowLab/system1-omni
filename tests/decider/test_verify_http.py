import unittest
from verify_http import concurrency_request

class CorpusSelection(unittest.TestCase):
    def test_short_custom_corpus_uses_first_nonempty_request(self):
        request = {"state":"long", "questions":{"q":{}}}
        self.assertIs(concurrency_request([{"name":"empty", "request":{"questions":{}}}, {"name":"long_shared", "request":request}]), request)
    def test_named_mixed_case_is_preferred_regardless_of_position(self):
        request = {"questions":{"mixed":{}}}
        self.assertIs(concurrency_request([{"name":"other", "request":{"questions":{"q":{}}}}, {"name":"mixed", "request":request}]), request)
    def test_all_empty_and_absent_corpora_are_explicit(self):
        request = {"questions":{}}
        self.assertIs(concurrency_request([{"name":"empty", "request":request}]), request)
        with self.assertRaises(ValueError):
            concurrency_request([])

if __name__ == "__main__":
    unittest.main()
