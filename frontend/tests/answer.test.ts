import { describe, expect, it } from "vitest";
import { answerResponseSchema } from "../lib/answer";
import { fullResponse } from "./fixtures";

describe("response validation", () => {
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
