import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnswerResult } from "../app/result";
import { fullResponse } from "./fixtures";

describe("grounded result presentation", () => {
  it("renders each section in order with backend figures and visible caveats", () => {
    const response = fullResponse();
    render(<AnswerResult response={response} />);
    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual([
      "Answer",
      "Metrics",
      "Findings",
      "Evidence",
      "Limitations",
      "Warnings",
    ]);
    const answer = screen.getByRole("region", { name: "Answer" });
    expect(within(answer).getByText(/A reviewer describes/).textContent).toBe(
      response.answer,
    );
    const metrics = screen.getByRole("region", { name: "Metrics" });
    expect(within(metrics).getAllByText("23")[0]).toBeVisible();
    expect(within(metrics).getAllByText("17")[0]).toBeVisible();
    expect(within(metrics).getAllByText("2.75★")[0]).toBeVisible();
    expect(
      within(screen.getByRole("region", { name: "Findings" })).getByText(
        "[E1]",
      ),
    ).toBeVisible();
    expect(screen.getByText(response.findings[0].claim)).toBeVisible();
    expect(screen.getByText(response.limitations[0])).toBeVisible();
    expect(screen.getByText(response.warnings[0])).toBeVisible();
    expect(
      screen.getByText("View trace").closest("details"),
    ).not.toHaveAttribute("open");
  });

  it("handles an empty metrics object without fabricated numbers", () => {
    render(<AnswerResult response={{ ...fullResponse(), metrics: {} }} />);
    expect(
      screen.getByText("No metrics were provided for this answer."),
    ).toBeVisible();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
  });

  it("renders string-key rating buckets in ascending order including zero", () => {
    render(<AnswerResult response={fullResponse()} />);
    const table = screen.getByRole("table", {
      name: "Rating distribution · reviews",
    });
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((cell) => cell.textContent),
    ).toEqual(["1★", "2★", "3★", "4★", "5★"]);
    expect(
      within(table)
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual(["7", "0", "8", "4", "4"]);
  });

  it("shows only the first three evidence cards and expands and collapses the rest", () => {
    render(<AnswerResult response={fullResponse()} />);
    expect(
      screen.getAllByRole("article", { name: /^Evidence E/ }),
    ).toHaveLength(3);
    expect(
      screen.queryByRole("article", { name: "Evidence E4" }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show more evidence" }));
    expect(
      screen.getAllByRole("article", { name: /^Evidence E/ }),
    ).toHaveLength(4);
    expect(
      screen.getByRole("button", { name: "Show less evidence" }),
    ).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(screen.getByRole("button", { name: "Show less evidence" }));
    expect(
      screen.getAllByRole("article", { name: /^Evidence E/ }),
    ).toHaveLength(3);
  });

  it("preserves excerpt whitespace, punctuation and Unicode without substituting the full body", () => {
    const response = fullResponse();
    render(<AnswerResult response={response} />);
    const card = screen.getByRole("article", { name: "Evidence E1" });
    expect(card.querySelector("blockquote")?.textContent).toBe(
      response.evidence[0].excerpt,
    );
    expect(screen.queryByText(/Additional body text/)).not.toBeInTheDocument();
    expect(within(card).getByLabelText("3 out of 5 stars")).toBeVisible();
    expect(within(card).getByText("Jan 9, 2026")).toHaveAttribute(
      "datetime",
      "2026-01-09T00:00:00Z",
    );
  });

  it("renders withheld narratives and empty findings and evidence as supplied", () => {
    render(
      <AnswerResult
        response={{
          ...fullResponse(),
          answer:
            "The narrative was withheld because support was insufficient.",
          findings: [],
          evidence: [],
          metrics: {},
          trace: { ...fullResponse().trace, synthesis_skipped: true },
        }}
      />,
    );
    expect(
      screen.getByText(
        "The narrative was withheld because support was insufficient.",
      ),
    ).toBeVisible();
    expect(screen.getByText("No findings were provided.")).toBeVisible();
    expect(screen.getByText("No review evidence was provided.")).toBeVisible();
    expect(
      screen.queryByRole("button", { name: /more evidence/ }),
    ).not.toBeInTheDocument();
  });

  it("renders nullable averages and date ranges without inventing zero values", () => {
    const response = fullResponse();
    render(
      <AnswerResult
        response={{
          ...response,
          metrics: {
            totals: { ...response.metrics.totals, overall_avg_rating: null },
            apps: [
              {
                ...response.metrics.apps[0],
                avg_rating: null,
                oldest_review_at: null,
                newest_review_at: null,
              },
            ],
          },
        }}
      />,
    );
    expect(
      screen.getAllByText("Not available", { selector: "dd" })[0],
    ).toBeVisible();
    fireEvent.click(screen.getByText("View metrics by app"));
    expect(
      screen.getAllByText("Not available", { selector: "dd" }),
    ).toHaveLength(2);
    expect(screen.queryByText(/Invalid Date|NaN/)).not.toBeInTheDocument();
  });

  it("renders HTML-like content as plain text", () => {
    const response = fullResponse();
    response.answer = '<script>alert("sample")</script> [E1]';
    render(<AnswerResult response={response} />);
    expect(screen.getByText(response.answer)).toBeVisible();
    expect(document.querySelector("script")).toBeNull();
  });
});
