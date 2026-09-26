# Backend response contract sample

`answer-response-sanitized.json` is derived from the real backend `AnswerResponse`
captured as `/tmp/answer-response-sample.json` and read during the initial frontend
milestone. That temporary file was no longer present during this QA pass, so the
source used here is the complete captured payload retained in the earlier task
conversation, not a new response synthesized from the frontend TypeScript types.

The sanitization retains every field name, JSON type, nesting level, array length,
null position, finding kind and citation relationship from that capture. It also
retains the numeric metrics, string rating bucket keys, review ratings, timestamps,
versions, similarity values, trace models/providers and booleans. The capture has
12 findings, 10 evidence entries, one app aggregate, one limitation and no warnings.

Every review title, body and excerpt was replaced with synthetic prose. The answer,
finding claims, question, semantic query and limitation were also replaced so that
quoted review text could not remain in those fields. App identifiers/names, review
identifiers and the query UUID were replaced with synthetic identifiers. No real
review text is included. The long fourth review still has a 500-character excerpt;
the other excerpts equal their synthetic review bodies.

The contract test parses this independent, static JSON and compares the entire
result for exact equality. It catches incompatible validation changes as well as
accidental field stripping. It requires no backend, temporary capture, network,
database or credentials at test time. Like any captured contract test, it pins this
recorded response shape, not every possible future backend payload.

The separate `../fixtures.ts` file is an authored synthetic rendering fixture. Its
findings all cite existing evidence, and its averages, buckets, matched counts,
excerpt boundaries and active provider trace are internally consistent. Tests that
deliberately mutate it into malformed or nullable defensive cases remain separate.
