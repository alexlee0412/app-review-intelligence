"use client";

import { useRef, useState, type FormEvent } from "react";
import { askReviews } from "../lib/ask";
import type { AnswerResponse } from "../lib/answer";
import { AnswerResult } from "./result";
import styles from "./page.module.css";

const examples = [
  "What problems are users reporting about EPIK?",
  "What are users complaining about regarding subscriptions?",
  "What do users dislike about pricing or paywalls?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const inFlight = useRef(false);
  const input = useRef<HTMLInputElement>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inFlight.current || !question.trim()) return;
    inFlight.current = true;
    setAsked(question.trim());
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const result = await askReviews(question.trim());
      if (result.ok) setAnswer(result.answer);
      else setError(result.message);
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1>App Review Intelligence</h1>
        <p className={styles.subtitle}>
          Ask what users are saying. Every figure is computed from the reviews,
          and every claim cites the ones it rests on.
        </p>
      </header>
      <section className={styles.askPanel} aria-label="Ask about reviews">
        <form onSubmit={submit}>
          <label htmlFor="question" className={styles.label}>
            Your question
          </label>
          <div className={styles.inputRow}>
            <input
              id="question"
              ref={input}
              value={question}
              disabled={loading}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What would you like to know?"
              required
              autoComplete="off"
            />
            <button
              className={styles.primary}
              type="submit"
              disabled={loading || !question.trim()}
            >
              {loading ? "Analyzing…" : error ? "Try again" : "Ask"}
            </button>
          </div>
        </form>
        <div className={styles.examples}>
          <span>Try asking</span>
          {examples.map((example) => (
            <button
              type="button"
              key={example}
              disabled={loading}
              onClick={() => {
                setQuestion(example);
                input.current?.focus();
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </section>
      <div role="status" aria-live="polite" aria-atomic="true">
        {loading && (
          <div className={styles.loading}>
            <strong>Analyzing app reviews…</strong>
            <p className={styles.loadingQuestion}>{asked}</p>
            <p>
              Reviews are being counted, matching passages retrieved, and the
              answer checked against them. This can take a little while.
            </p>
            <div className={styles.progress} aria-hidden="true" />
          </div>
        )}
        {answer && (
          <span className={styles.srOnly}>
            Analysis ready. Read the answer and supporting evidence below.
          </span>
        )}
      </div>
      {error && (
        <div role="alert" className={styles.notice}>
          <h2>Unable to complete analysis</h2>
          <p>{error}</p>
        </div>
      )}
      {answer && <AnswerResult key={answer.query_run_id} response={answer} />}
      {!answer && !loading && !error && (
        <p className={styles.empty}>
          Ask a question to see the answer, the figures behind it, and the
          reviews each claim cites.
        </p>
      )}
    </main>
  );
}
