import { z } from "zod";

const count = z.number().int().nonnegative();
const average = z.number().min(1).max(5).nullable();
const timestamp = z.iso.datetime({ offset: true });
const evidenceId = z.string().regex(/^E[1-9][0-9]*$/);
export const ratingBuckets = ["1", "2", "3", "4", "5"] as const;
const distribution = z.object({
  "1": count,
  "2": count,
  "3": count,
  "4": count,
  "5": count,
});
const filters = z.object({
  app_ids: z.array(z.string()).nullable(),
  countries: z.array(z.string()).nullable(),
  ratings: z.array(z.number().int().min(1).max(5)).nullable(),
  date_from: timestamp.nullable(),
  date_to: timestamp.nullable(),
  min_similarity: z.number().min(0).max(1).nullable(),
});

export const answerResponseSchema = z.object({
  query_run_id: z.uuid(),
  question: z.string(),
  answer: z.string(),
  findings: z.array(
    z.object({
      claim: z.string(),
      evidence_ids: z.array(evidenceId).min(1),
      kind: z.enum(["observed", "computed", "interpretation"]),
    }),
  ),
  metrics: z.union([
    z.object({
      totals: z.object({
        total_reviews: count,
        total_matched: count,
        apps_with_matches: count,
        overall_avg_rating: average,
        rating_distribution: distribution,
      }),
      apps: z.array(
        z.object({
          app_id: z.string(),
          app_name: z.string(),
          review_count: count,
          matched_count: count,
          avg_rating: average,
          rating_distribution: distribution,
          oldest_review_at: timestamp.nullable(),
          newest_review_at: timestamp.nullable(),
        }),
      ),
    }),
    z.strictObject({}),
  ]),
  evidence: z.array(
    z.object({
      evidence_id: evidenceId,
      excerpt: z.string(),
      review: z.object({
        review_id: z.string(),
        app_id: z.string(),
        app_name: z.string(),
        rating: z.number().int().min(1).max(5),
        country: z.string(),
        created_at: timestamp,
        title: z.string().nullable(),
        body: z.string(),
        version: z.string().nullable(),
        similarity: z.number().min(0).max(1),
      }),
    }),
  ),
  limitations: z.array(z.string()),
  warnings: z.array(z.string()),
  trace: z.object({
    intent: z.string(),
    applied_filters: filters,
    semantic_query: z.string().nullable(),
    total_candidates: count,
    evidence_count: count,
    aggregates_computed: z.array(z.string()),
    planner_model: z.string().nullable(),
    synthesizer_model: z.string().nullable(),
    llm_provider: z.string().nullable(),
    llm_is_production_grade: z.boolean().nullable(),
    embedding_provider: z.string().nullable(),
    embedding_is_production_grade: z.boolean().nullable(),
    synthesis_skipped: z.boolean(),
  }),
});

export type AnswerResponse = z.infer<typeof answerResponseSchema>;
export type RatingDistribution = z.infer<typeof distribution>;
