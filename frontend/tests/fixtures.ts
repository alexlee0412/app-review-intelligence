import type { AnswerResponse } from "../lib/answer";

// All review text and values in this fixture are fictional test data.
const syntheticResponse = {
  query_run_id: "12345678-1234-4234-8234-123456789abc",
  question: "What do users think of Sample Camera?",
  answer:
    "A reviewer describes an export problem. [E1]\n\n- Another asks for a preview. [E2]\n- Editing feedback is mixed. [E3][E4]",
  findings: [
    {
      claim: "A reviewer reports that export pauses.",
      evidence_ids: ["E1"],
      kind: "observed",
    },
    {
      claim: "The sample dataset contains 23 reviews.",
      evidence_ids: ["E2"],
      kind: "computed",
    },
    {
      claim: "Editing controls may interrupt the workflow.",
      evidence_ids: ["E3", "E4"],
      kind: "interpretation",
    },
  ],
  metrics: {
    totals: {
      total_reviews: 23,
      total_matched: 4,
      apps_with_matches: 1,
      overall_avg_rating: 2.9130434782608696,
      rating_distribution: { "1": 7, "2": 0, "3": 8, "4": 4, "5": 4 },
    },
    apps: [
      {
        app_id: "sample-camera",
        app_name: "Sample Camera",
        review_count: 23,
        matched_count: 4,
        avg_rating: 2.9130434782608696,
        rating_distribution: { "1": 7, "2": 0, "3": 8, "4": 4, "5": 4 },
        oldest_review_at: "2026-01-01T00:00:00Z",
        newest_review_at: "2026-01-09T00:00:00Z",
      },
    ],
  },
  evidence: [
    Array.from(
      "Export pauses when I select a frame.\nI tried again — same result!  Please keep my edits. 🧩" +
        " Synthetic export feedback.".repeat(30),
    )
      .slice(0, 500)
      .join(""),
    "A preview before saving would help me choose a border.",
    "The crop control is easy to find. The rotation control feels awkward.",
    "I like the color controls, but the toolbar takes up too much space.",
  ].map((excerpt, index) => ({
    evidence_id: `E${index + 1}`,
    excerpt,
    review: {
      review_id: `synthetic-review-${index + 1}`,
      app_id: "sample-camera",
      app_name: "Sample Camera",
      rating: 3,
      country: "US",
      created_at: "2026-01-09T00:00:00Z",
      title: null,
      body:
        index === 0
          ? `${excerpt}\nAdditional body text is deliberately excluded from the excerpt.`
          : excerpt,
      version: null,
      similarity: 0.5,
    },
  })),
  limitations: ["These fictional reviews cover only the sample dataset."],
  warnings: [
    "The non-production answer provider ('synthetic-provider') was used; generated language is not production-grade.",
    "The non-semantic development embedding provider ('synthetic-embedding') was used; retrieval scores and ordering are not meaningful.",
  ],
  trace: {
    intent: "review_summary",
    applied_filters: {
      app_ids: ["sample-camera"],
      countries: ["US"],
      ratings: null,
      date_from: null,
      date_to: null,
      min_similarity: null,
    },
    semantic_query: "editing feedback",
    total_candidates: 17,
    evidence_count: 4,
    aggregates_computed: ["review_count", "avg_rating"],
    planner_model: "synthetic-planner",
    synthesizer_model: "synthetic-synthesizer",
    llm_provider: "synthetic-provider",
    llm_is_production_grade: false,
    embedding_provider: "synthetic-embedding",
    embedding_is_production_grade: false,
    synthesis_skipped: false,
  },
} satisfies AnswerResponse;

export function fullResponse() {
  return structuredClone(syntheticResponse);
}
