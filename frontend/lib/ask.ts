import { answerResponseSchema, type AnswerResponse } from "./answer";

export const errorMessages = {
  configuration:
    "The review service is not configured. Check the local setup, then try again.",
  network:
    "We couldn’t reach the review service. Check your connection and try again.",
  question:
    "We couldn’t process that question. Please rephrase it and try again.",
  unavailable:
    "The review service is temporarily unavailable. Please try again.",
  response: "We couldn’t read the review service’s response. Please try again.",
  other: "Something went wrong while analyzing reviews. Please try again.",
} as const;

type AskResult =
  { ok: true; answer: AnswerResponse } | { ok: false; message: string };

export async function askReviews(question: string): Promise<AskResult> {
  let endpoint: URL;
  try {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
    if (!base) return { ok: false, message: errorMessages.configuration };
    endpoint = new URL(`${base.replace(/\/+$/, "")}/api/v1/reviews/ask`);
    if (
      !["http:", "https:"].includes(endpoint.protocol) ||
      endpoint.username ||
      endpoint.password ||
      endpoint.search ||
      endpoint.hash
    ) {
      return { ok: false, message: errorMessages.configuration };
    }
  } catch {
    return { ok: false, message: errorMessages.configuration };
  }

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      cache: "no-store",
      credentials: "omit",
    });
  } catch {
    return { ok: false, message: errorMessages.network };
  }

  if (!response.ok) {
    const message =
      response.status === 400 || response.status === 422
        ? errorMessages.question
        : response.status === 503
          ? errorMessages.unavailable
          : errorMessages.other;
    return { ok: false, message };
  }
  try {
    const parsed = answerResponseSchema.safeParse(await response.json());
    return parsed.success
      ? { ok: true, answer: parsed.data }
      : { ok: false, message: errorMessages.response };
  } catch {
    return { ok: false, message: errorMessages.response };
  }
}
