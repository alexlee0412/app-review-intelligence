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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const inFlight = useRef(false);
  const input = useRef<HTMLInputElement>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inFlight.current || !question.trim()) return;
    inFlight.current = true;
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
        <p className={styles.eyebrow}>REVIEWS, WITH EVIDENCE</p>
        <h1>App Review Intelligence</h1>
        <p className={styles.subtitle}>
          Ask what users are saying across app reviews.
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
            <span className={styles.spinner} aria-hidden="true" />
            <div>
              <strong>Analyzing app reviews…</strong>
              <p>This can take a little while. Your answer will appear here.</p>
            </div>
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
          Start with a question. Explore the answer, its metrics, and the
          reviews behind it.
        </p>
      )}
    </main>
  );
}
