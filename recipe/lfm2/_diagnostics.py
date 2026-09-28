"""Pinned diagnostic schemas and cases for the LFM2 recipe checks.

Fixture only, copied from notnotsamuel/LFM2.5-350M-RLCD revision
deb589d803d141cabd158ef55f6617b128529f36 (``rlcd/tasks.py`` and
``rlcd/stress_tasks.py``, MIT, Copyright (c) 2026 notnotsamuel). The labels are
hand-authored diagnostics, not a population accuracy benchmark. The fixtures are kept
in-tree so the workloads can be reviewed before fetching the reference package.
"""


def schema(**fields):
    return {"type": "object", "properties": fields, "required": list(fields),
            "additionalProperties": False}


def enum(description, *values):
    return {"type": "string", "description": description, "enum": list(values)}


def boolean(description):
    return {"type": "boolean", "description": description}


SUPPORT = schema(
    topic=enum("Main issue", "billing", "technical", "shipping"),
    urgent=boolean("True only if immediate action is explicitly needed"),
    refund=boolean("Whether a refund is requested"),
)
SENTIMENT = schema(
    sentiment=enum("Overall sentiment", "positive", "negative", "neutral"),
    language=enum("Language of the text", "English", "French", "Spanish"),
    question=boolean("Whether the text asks a question"),
)
ROUTING = schema(
    route=enum("Copy the exact route named in the text", "north east", "north west",
               "south east", "south west"),
    service=enum("Requested delivery service", "standard", "express", "express plus"),
    insured=boolean("Whether insurance is requested"),
)

CASES = [
    ("support-1", SUPPORT, "I was charged twice. Please refund the duplicate charge. This can wait until next week.", {"topic": "billing", "urgent": False, "refund": True}),
    ("support-2", SUPPORT, "The application crashes on startup. We need immediate action; our entire team is blocked. No refund needed.", {"topic": "technical", "urgent": True, "refund": False}),
    ("support-3", SUPPORT, "Where is my parcel? There is no hurry and I do not want a refund.", {"topic": "shipping", "urgent": False, "refund": False}),
    ("support-4", SUPPORT, "My parcel has not arrived. Please refund the shipping fee immediately; I need immediate action.", {"topic": "shipping", "urgent": True, "refund": True}),
    ("sentiment-1", SENTIMENT, "This product is wonderful. I love it.", {"sentiment": "positive", "language": "English", "question": False}),
    ("sentiment-2", SENTIMENT, "Ce produit est horrible. Pourquoi est-il si mauvais ?", {"sentiment": "negative", "language": "French", "question": True}),
    ("sentiment-3", SENTIMENT, "El paquete contiene tres piezas.", {"sentiment": "neutral", "language": "Spanish", "question": False}),
    ("sentiment-4", SENTIMENT, "Does the box contain three parts?", {"sentiment": "neutral", "language": "English", "question": True}),
    ("routing-1", ROUTING, "Route: north east. Service: express plus. Insurance requested.", {"route": "north east", "service": "express plus", "insured": True}),
    ("routing-2", ROUTING, "Route: north west. Service: express. No insurance.", {"route": "north west", "service": "express", "insured": False}),
    ("routing-3", ROUTING, "Route: south east. Service: standard. Insurance requested.", {"route": "south east", "service": "standard", "insured": True}),
    ("routing-4", ROUTING, "Route: south west. Service: express plus. No insurance.", {"route": "south west", "service": "express plus", "insured": False}),
]

STRESS_CASES = []
for _cardinality in [64, 255]:
    _choices = ["tariff_%03d" % index for index in range(_cardinality)]
    _schema = schema(tariff=enum("Copy the exact tariff code in the text", *_choices))
    STRESS_CASES.append(("enum-%d" % _cardinality, _schema,
                         "The assigned tariff code is %s." % _choices[-2],
                         {"tariff": _choices[-2]}))
for _count in [12, 28]:
    _fields = {"flag_%02d" % index: boolean(
        "True if item %02d is enabled; false if disabled" % index) for index in range(_count)}
    _expected = {name: index % 2 == 0 for index, name in enumerate(_fields)}
    _context = "\n".join("Item %02d is %s." % (
        index, "enabled" if index % 2 == 0 else "disabled") for index in range(_count))
    STRESS_CASES.append(("fields-%d" % _count, schema(**_fields), _context, _expected))
_signal = schema(signal=enum("Copy the final signal color", "red", "green", "blue"))
_context = "Archive entries follow.\n" + "An old record was checked and filed.\n" * 160 \
    + "\nFinal signal color: blue."
STRESS_CASES.append(("long-context", _signal, _context, {"signal": "blue"}))
