"use client";

import { useState } from "react";
import {
  ratingBuckets,
  type AnswerResponse,
  type RatingDistribution,
} from "../lib/answer";
import styles from "./page.module.css";

function Distribution({
  values,
  label,
}: {
  values: RatingDistribution;
  label: string;
}) {
  return (
    <table className={styles.distribution}>
      <caption>{label}</caption>
      <thead>
        <tr>
          {ratingBuckets.map((rating) => (
            <th scope="col" key={rating}>
              {rating}★
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          {ratingBuckets.map((rating) => (
            <td key={rating}>{values[rating]}</td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

function ReviewDate({ value }: { value: string }) {
  return (
    <time dateTime={value}>
      {new Intl.DateTimeFormat("en", {
        year: "numeric",
        month: "short",
        day: "numeric",
        timeZone: "UTC",
      }).format(new Date(value))}
    </time>
  );
}

export function AnswerResult({ response }: { response: AnswerResponse }) {
  const [expanded, setExpanded] = useState(false);
  const { metrics } = response;
  const visibleEvidence = expanded
    ? response.evidence
    : response.evidence.slice(0, 3);

  return (
    <article className={styles.results} aria-label="Review analysis">
      <section className={styles.section} aria-labelledby="answer-heading">
        <p className={styles.questionAsked}>{response.question}</p>
        <h2 id="answer-heading">Answer</h2>
        <div className={styles.verbatim}>{response.answer}</div>
      </section>
      <section className={styles.metrics} aria-labelledby="metrics-heading">
        <h2 id="metrics-heading">Metrics</h2>
        {"totals" in metrics ? (
          <>
            <p className={styles.sectionNote}>Computed from review data</p>
            <dl className={styles.figures}>
              <div>
                <dt>Reviews analyzed</dt>
                <dd>{metrics.totals.total_reviews}</dd>
              </div>
              <div>
                <dt>Matched reviews</dt>
                <dd>{metrics.totals.total_matched}</dd>
              </div>
              <div>
                <dt>Apps with matches</dt>
                <dd>{metrics.totals.apps_with_matches}</dd>
              </div>
              <div>
                <dt>Average rating</dt>
                <dd>
                  {metrics.totals.overall_avg_rating === null
                    ? "Not available"
                    : `${metrics.totals.overall_avg_rating.toFixed(2)}★`}
                </dd>
              </div>
            </dl>
            <Distribution
              values={metrics.totals.rating_distribution}
              label="Rating distribution · reviews"
            />
            {metrics.apps.length > 0 && (
              <details className={styles.appMetrics}>
                <summary>View metrics by app</summary>
                {metrics.apps.map((app) => (
                  <section key={app.app_id} className={styles.appMetric}>
                    <h3>{app.app_name}</h3>
                    <dl className={styles.figures}>
                      <div>
                        <dt>Reviews analyzed</dt>
                        <dd>{app.review_count}</dd>
                      </div>
                      <div>
                        <dt>Matched reviews</dt>
                        <dd>{app.matched_count}</dd>
                      </div>
                      <div>
                        <dt>Average rating</dt>
                        <dd>
                          {app.avg_rating === null
                            ? "Not available"
                            : `${app.avg_rating.toFixed(2)}★`}
                        </dd>
                      </div>
                    </dl>
                    <Distribution
                      values={app.rating_distribution}
                      label={`${app.app_name} rating distribution · reviews`}
                    />
                    <p>
                      Oldest review:{" "}
                      {app.oldest_review_at ? (
                        <ReviewDate value={app.oldest_review_at} />
                      ) : (
                        "Not available"
                      )}
                      <br />
                      Newest review:{" "}
                      {app.newest_review_at ? (
                        <ReviewDate value={app.newest_review_at} />
                      ) : (
                        "Not available"
                      )}
                    </p>
                  </section>
                ))}
              </details>
            )}
          </>
        ) : (
          <p>No metrics were provided for this answer.</p>
        )}
      </section>
      <section className={styles.section} aria-labelledby="findings-heading">
        <h2 id="findings-heading">Findings</h2>
        {response.findings.length ? (
          <ul className={styles.findings}>
            {response.findings.map((finding, index) => (
              <li key={index}>
                <p className={styles.verbatim}>{finding.claim}</p>
                <div className={styles.chips}>
                  <span className={`${styles.chip} ${styles[finding.kind]}`}>
                    {finding.kind}
                  </span>
                  {finding.evidence_ids.map((id, index) => (
                    <span className={styles.chip} key={`${id}-${index}`}>
                      [{id}]
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p>No findings were provided.</p>
        )}
      </section>
      <section className={styles.section} aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence</h2>
        <div id="evidence-cards" className={styles.evidenceCards}>
          {visibleEvidence.map((item) => (
            <article
              key={item.evidence_id}
              className={styles.evidenceCard}
              aria-label={`Evidence ${item.evidence_id}`}
            >
              <header>
                <span className={styles.chip}>[{item.evidence_id}]</span>
                <h3>{item.review.app_name}</h3>
              </header>
              <p className={styles.reviewMeta}>
                <span aria-label={`${item.review.rating} out of 5 stars`}>
                  {item.review.rating}★
                </span>
                <span>{item.review.country}</span>
                <ReviewDate value={item.review.created_at} />
              </p>
              <blockquote className={styles.verbatim}>
                {item.excerpt}
              </blockquote>
            </article>
          ))}
        </div>
        {response.evidence.length === 0 && (
          <p>No review evidence was provided.</p>
        )}
        {response.evidence.length > 3 && (
          <button
            type="button"
            className={styles.secondary}
            aria-expanded={expanded}
            aria-controls="evidence-cards"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Show less evidence" : "Show more evidence"}
          </button>
        )}
      </section>
      {response.limitations.length > 0 && (
        <section
          className={styles.notice}
          aria-labelledby="limitations-heading"
        >
          <h2 id="limitations-heading">Limitations</h2>
          <ul>
            {response.limitations.map((limitation, index) => (
              <li className={styles.verbatim} key={index}>
                {limitation}
              </li>
            ))}
          </ul>
        </section>
      )}
      {response.warnings.length > 0 && (
        <section className={styles.notice} aria-labelledby="warnings-heading">
          <h2 id="warnings-heading">Warnings</h2>
          <ul>
            {response.warnings.map((warning, index) => (
              <li className={styles.verbatim} key={index}>
                {warning}
              </li>
            ))}
          </ul>
        </section>
      )}
      <details className={styles.trace}>
        <summary>View trace</summary>
        <pre>
          {JSON.stringify(
            { query_run_id: response.query_run_id, ...response.trace },
            null,
            2,
          )}
        </pre>
      </details>
    </article>
  );
}
