"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  ratingBuckets,
  type AnswerResponse,
  type RatingDistribution,
} from "../lib/answer";
import styles from "./page.module.css";

const visibleByDefault = 3;
// Splits answer prose on its citation marks while keeping the marks in place,
// so the rendered text stays character-for-character what the backend sent.
const citationMark = /(\[E[1-9][0-9]*\])/g;

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

function Citation({
  id,
  selected,
  onSelect,
}: {
  id: string;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      type="button"
      className={styles.citation}
      aria-current={selected ? "true" : undefined}
      aria-label={`Show the review cited as ${id}`}
      onClick={() => onSelect(id)}
    >
      [{id}]
    </button>
  );
}

// Citations resolve to evidence the backend actually supplied; an unresolved
// mark stays plain text rather than becoming a control that leads nowhere.
function withCitations(
  text: string,
  resolvable: ReadonlySet<string>,
  selected: string | null,
  onSelect: (id: string) => void,
): ReactNode[] {
  return text.split(citationMark).map((part, index) => {
    const id = /^\[(E[1-9][0-9]*)\]$/.exec(part)?.[1];
    return id && resolvable.has(id) ? (
      <Citation
        key={index}
        id={id}
        selected={id === selected}
        onSelect={onSelect}
      />
    ) : (
      part
    );
  });
}

function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function AnswerResult({ response }: { response: AnswerResponse }) {
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [request, setRequest] = useState<{ id: string; seq: number } | null>(
    null,
  );
  const cards = useRef(new Map<string, HTMLElement>());
  const { metrics } = response;

  const order = response.evidence.map((item) => item.evidence_id);
  const resolvable = new Set(order);
  const visibleEvidence = expanded
    ? response.evidence
    : response.evidence.slice(0, visibleByDefault);

  // Reveal the cited review wherever it sits: expand the list if the card is
  // collapsed, mark it, then move the page and keyboard focus to it.
  function selectEvidence(id: string) {
    const index = order.indexOf(id);
    if (index < 0) return;
    if (index >= visibleByDefault) setExpanded(true);
    setSelected(id);
    // The counter re-runs the reveal effect when the same citation is used twice.
    setRequest((previous) => ({ id, seq: (previous?.seq ?? 0) + 1 }));
  }

  function toggleEvidence() {
    const next = !expanded;
    setExpanded(next);
    if (!next && selected && order.indexOf(selected) >= visibleByDefault) {
      setSelected(null);
    }
  }

  useEffect(() => {
    if (!request) return;
    const card = cards.current.get(request.id);
    if (!card) return;
    card.scrollIntoView?.({ block: "center" });
    card.focus?.({ preventScroll: true });
  }, [request]);

  const limitCount = response.limitations.length;
  const warningCount = response.warnings.length;
  const limitParts = [
    limitCount > 0 && plural(limitCount, "limitation"),
    warningCount > 0 && plural(warningCount, "warning"),
  ].filter((part): part is string => Boolean(part));

  return (
    <article className={styles.report} aria-label="Review analysis">
      <div className={styles.sheet}>
        <section
          className={styles.answerSection}
          aria-labelledby="answer-heading"
        >
          <h2 id="answer-heading">Answer</h2>
          <span className={styles.provenance}>
            Written by the model from the evidence below
          </span>
          <p className={styles.questionAsked}>{response.question}</p>
          <div className={styles.answerBody}>
            {withCitations(
              response.answer,
              resolvable,
              selected,
              selectEvidence,
            )}
          </div>
          {limitParts.length > 0 && (
            <p className={styles.grounding}>
              <span>
                {limitParts.join(" and ")}{" "}
                {limitParts.length === 1 && limitCount + warningCount === 1
                  ? "applies"
                  : "apply"}{" "}
                to this answer.
              </span>
              <a href={limitCount > 0 ? "#limitations" : "#warnings"}>
                {limitCount + warningCount === 1 ? "Read it" : "Read them"}
              </a>
            </p>
          )}
        </section>
        <section className={styles.metrics} aria-labelledby="metrics-heading">
          <h2 id="metrics-heading">Metrics</h2>
          {"totals" in metrics ? (
            <>
              <span className={styles.provenance}>
                Measured from review data
              </span>
              <dl className={styles.figures}>
                <div className={styles.lead}>
                  <dt>Average rating</dt>
                  <dd
                    aria-label={
                      metrics.totals.overall_avg_rating === null
                        ? undefined
                        : `${metrics.totals.overall_avg_rating.toFixed(2)} out of 5 stars`
                    }
                  >
                    {metrics.totals.overall_avg_rating === null
                      ? "Not available"
                      : `${metrics.totals.overall_avg_rating.toFixed(2)}★`}
                  </dd>
                </div>
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
              </dl>
              <Distribution
                values={metrics.totals.rating_distribution}
                label="Rating distribution across all reviews"
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
                          <dd
                            aria-label={
                              app.avg_rating === null
                                ? undefined
                                : `${app.avg_rating.toFixed(2)} out of 5 stars`
                            }
                          >
                            {app.avg_rating === null
                              ? "Not available"
                              : `${app.avg_rating.toFixed(2)}★`}
                          </dd>
                        </div>
                      </dl>
                      <Distribution
                        values={app.rating_distribution}
                        label={`Rating distribution for ${app.app_name}`}
                      />
                      <p className={styles.appDates}>
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
            <p className={styles.sectionNote}>
              No metrics were provided for this answer.
            </p>
          )}
        </section>
        <section className={styles.findings} aria-labelledby="findings-heading">
          <h2 id="findings-heading">Findings</h2>
          <span className={styles.provenance}>
            Each claim carries the reviews it rests on
          </span>
          {response.findings.length ? (
            <ul className={styles.findingList}>
              {response.findings.map((finding, index) => (
                <li key={index}>
                  <span className={`${styles.kind} ${styles[finding.kind]}`}>
                    {finding.kind}
                  </span>
                  <p>
                    {finding.claim}{" "}
                    {finding.evidence_ids.map((id, position) => (
                      <Citation
                        key={`${id}-${position}`}
                        id={id}
                        selected={id === selected}
                        onSelect={selectEvidence}
                      />
                    ))}
                  </p>
                </li>
              ))}
            </ul>
          ) : (
            <p className={styles.sectionNote}>No findings were provided.</p>
          )}
        </section>
      </div>
      <section className={styles.evidence} aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence</h2>
        <span className={styles.provenance}>
          Quoted verbatim from the reviews themselves
        </span>
        <div id="evidence-cards" className={styles.evidenceCards}>
          {visibleEvidence.map((item) => (
            <article
              key={item.evidence_id}
              id={`evidence-${item.evidence_id}`}
              className={styles.evidenceCard}
              aria-label={`Evidence ${item.evidence_id}`}
              aria-current={
                item.evidence_id === selected ? "true" : undefined
              }
              tabIndex={-1}
              ref={(node) => {
                if (node) cards.current.set(item.evidence_id, node);
                else cards.current.delete(item.evidence_id);
              }}
            >
              <header>
                <span className={styles.evidenceId}>[{item.evidence_id}]</span>
                <h3>{item.review.app_name}</h3>
                <p className={styles.reviewMeta}>
                  <span aria-label={`${item.review.rating} out of 5 stars`}>
                    {item.review.rating}★
                  </span>
                  <span>{item.review.country}</span>
                  <ReviewDate value={item.review.created_at} />
                </p>
              </header>
              <blockquote className={styles.excerpt}>
                {item.excerpt}
              </blockquote>
            </article>
          ))}
        </div>
        {response.evidence.length === 0 && (
          <p className={styles.sectionNote}>
            No review evidence was provided.
          </p>
        )}
        {response.evidence.length > visibleByDefault && (
          <button
            type="button"
            className={styles.secondary}
            aria-expanded={expanded}
            aria-controls="evidence-cards"
            onClick={toggleEvidence}
          >
            {expanded ? "Show less evidence" : "Show more evidence"}
          </button>
        )}
      </section>
      {response.limitations.length > 0 && (
        <section
          id="limitations"
          className={styles.limits}
          aria-labelledby="limitations-heading"
        >
          <h2 id="limitations-heading">Limitations</h2>
          <ul>
            {response.limitations.map((limitation, index) => (
              <li key={index}>{limitation}</li>
            ))}
          </ul>
        </section>
      )}
      {response.warnings.length > 0 && (
        <section
          id="warnings"
          className={styles.limits}
          aria-labelledby="warnings-heading"
        >
          <h2 id="warnings-heading">Warnings</h2>
          <ul>
            {response.warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
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
