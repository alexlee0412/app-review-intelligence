import { describe, expect, it } from "vitest";
import { answerResponseSchema } from "../lib/answer";
import { fullResponse } from "./fixtures";
import capturedResponse from "./fixtures/answer-response-sanitized.json";

describe("response validation", () => {
  it("accepts the sanitized backend capture without dropping or reshaping fields", () => {
    // This frozen JSON comes from the captured backend payload, not the TS schema.
    // Equality also detects fields silently stripped by a drifting Zod object.
    expect(answerResponseSchema.parse(capturedResponse)).toStrictEqual(
      capturedResponse,
    );
  });

  it("accepts the synthetic contract and preserves its data", () => {
    const response = fullResponse();
    expect(answerResponseSchema.parse(response)).toEqual(response);
  });

  it("accepts empty metrics", () => {
    expect(
      answerResponseSchema.safeParse({ ...fullResponse(), metrics: {} })
        .success,
    ).toBe(true);
  });

  it.each(["observed", "computed", "interpretation"])(
    "rejects an uncited %s finding",
    (kind) => {
      const response = fullResponse();
      expect(
        answerResponseSchema.safeParse({
          ...response,
          findings: [{ claim: "Uncited claim", kind, evidence_ids: [] }],
        }).success,
      ).toBe(false);
    },
  );

  it("keeps synthetic findings, aggregates, excerpts and trace internally consistent", () => {
    const response = fullResponse();
    const ids = new Set(response.evidence.map((item) => item.evidence_id));
    for (const finding of response.findings) {
      expect(finding.evidence_ids.length).toBeGreaterThan(0);
      for (const id of finding.evidence_ids) expect(ids.has(id)).toBe(true);
    }
    const app = response.metrics.apps[0];
    const buckets = Object.entries(app.rating_distribution);
    expect(buckets.reduce((sum, [, count]) => sum + count, 0)).toBe(
      app.review_count,
    );
    const average =
      buckets.reduce(
        (sum, [rating, count]) => sum + Number(rating) * count,
        0,
      ) / app.review_count;
    expect(app.avg_rating).toBe(average);
    expect(response.metrics.totals.overall_avg_rating).toBe(average);
    expect(response.metrics.totals.total_reviews).toBe(app.review_count);
    expect(response.metrics.totals.rating_distribution).toEqual(
      app.rating_distribution,
    );
    expect(app.matched_count).toBe(response.evidence.length);
    expect(response.metrics.totals.total_matched).toBe(app.matched_count);
    expect(response.trace.evidence_count).toBe(response.evidence.length);
    expect(response.trace.synthesizer_model).not.toBeNull();
    for (const item of response.evidence) {
      // The backend slices Python strings by Unicode code point, not UTF-16 unit.
      expect(item.excerpt).toBe(
        Array.from(item.review.body).slice(0, 500).join(""),
      );
    }
  });

  it.each([
    null,
    {},
    { ...fullResponse(), query_run_id: "invalid" },
    { ...fullResponse(), warnings: "unexpected" },
    {
      ...fullResponse(),
      findings: [{ claim: "Example", evidence_ids: ["E0"], kind: "observed" }],
    },
    { ...fullResponse(), metrics: { unexpected: true } },
    {
      ...fullResponse(),
      metrics: {
        ...fullResponse().metrics,
        totals: { ...fullResponse().metrics.totals, total_reviews: -1 },
      },
    },
    {
      ...fullResponse(),
      metrics: {
        ...fullResponse().metrics,
        totals: {
          ...fullResponse().metrics.totals,
          overall_avg_rating: "2.75",
        },
      },
    },
    {
      ...fullResponse(),
      metrics: {
        ...fullResponse().metrics,
        totals: {
          ...fullResponse().metrics.totals,
          rating_distribution: { "1": 7, "3": 8, "4": 4, "5": 4 },
        },
      },
    },
    {
      ...fullResponse(),
      evidence: [
        {
          ...fullResponse().evidence[0],
          review: {
            ...fullResponse().evidence[0].review,
            created_at: "yesterday",
          },
        },
      ],
    },
  ])("rejects malformed data %#", (response) => {
    expect(answerResponseSchema.safeParse(response).success).toBe(false);
  });

  it("drops unknown response and trace fields before they can be displayed", () => {
    const parsed = answerResponseSchema.parse({
      ...fullResponse(),
      extra: "internal",
      trace: { ...fullResponse().trace, detail: "internal" },
    });
    expect(parsed).not.toHaveProperty("extra");
    expect(parsed.trace).not.toHaveProperty("detail");
  });
});
