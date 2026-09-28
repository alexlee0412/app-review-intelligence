import { readFileSync } from "node:fs";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AnswerResult } from "../app/result";
import styles from "../app/page.module.css";
import { fullResponse } from "./fixtures";

// Apply the production stylesheet with its module class names in jsdom.
const stylesheet = document.createElement("style");
beforeAll(() => {
  // jsdom does not resolve custom properties in computed styles. Use the actual
  // light-theme token values; keep all computed-style assertions unchanged.
  const theme = readFileSync("app/theme.css", "utf8").split("@media")[0];
  const tokens = new Map(
    Array.from(theme.matchAll(/(--[\w-]+):\s*([^;]+);/g), ([, name, value]) => [
      name,
      value,
    ]),
  );
  const css = readFileSync("app/page.module.css", "utf8").replace(
    /var\((--[\w-]+)\)/g,
    (value, name: string) => tokens.get(name) ?? value,
  );
  stylesheet.textContent = css.replace(
    /\.([a-zA-Z_][\w-]*)/g,
    (selector, name: string) => (styles[name] ? `.${styles[name]}` : selector),
  );
  document.head.appendChild(stylesheet);
});
afterAll(() => stylesheet.remove());

describe("grounded result presentation", () => {
  it("shows all finding kinds as distinct text tags in the supplied order", () => {
    const response = fullResponse();
    render(<AnswerResult response={response} />);
    const rows = within(
      screen.getByRole("region", { name: "Findings" }),
    ).getAllByRole("listitem");
    const kinds = ["observed", "computed", "interpretation"];
    expect(rows).toHaveLength(kinds.length);
    const backgrounds = rows.map((row, index) => {
      expect(
        within(row).getByText(response.findings[index].claim),
      ).toBeVisible();
      const tag = within(row).getByText(kinds[index], { exact: true });
      expect(tag).toBeVisible();
      for (const id of response.findings[index].evidence_ids) {
        expect(within(row).getByText(`[${id}]`, { exact: true })).toBeVisible();
      }
      return getComputedStyle(tag).backgroundColor;
    });
    expect(new Set(backgrounds).size).toBe(3);
  });

  it.each([
    [2.2857142857142856, "2.29★"],
    [2, "2.00★"],
  ])(
    "formats per-app average %s to two decimal places without changing data",
    (average, display) => {
      const response = fullResponse();
      response.metrics.apps[0].avg_rating = average;
      const originalMetrics = structuredClone(response.metrics);
      render(<AnswerResult response={response} />);
      const toggle = screen.getByText("View metrics by app");
      const panel = within(toggle.closest("details")!);
      expect(panel.getByText(display)).not.toBeVisible();
      fireEvent.click(toggle);
      expect(panel.getByText(display)).toBeVisible();
      expect(panel.getByText(display)).toHaveAttribute(
        "aria-label",
        `${average.toFixed(2)} out of 5 stars`,
      );
      expect(
        panel.queryByText(`${average}★`, { selector: "dd" }),
      ).not.toBeInTheDocument();
      expect(response.metrics).toEqual(originalMetrics);
    },
  );

  it("preserves multiline answer text with the production pre-wrap style", () => {
    const response = fullResponse();
    render(<AnswerResult response={response} />);
    const answer = within(
      screen.getByRole("region", { name: "Answer" }),
    ).getByText(/A reviewer describes/);
    expect(response.answer).toContain("\n\n- ");
    expect(answer.textContent).toBe(response.answer);
    expect(getComputedStyle(answer).whiteSpace).toBe("pre-wrap");
  });

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
    expect(within(metrics).getAllByText("4")[0]).toBeVisible();
    expect(within(metrics).getAllByText("2.91★")[0]).toBeVisible();
    expect(within(metrics).getAllByText("2.91★")[0]).toHaveAttribute(
      "aria-label",
      "2.91 out of 5 stars",
    );
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
      name: "Rating distribution across all reviews",
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
    for (const average of screen.getAllByText("Not available", {
      selector: "dd",
    })) {
      expect(average).not.toHaveAttribute("aria-label");
    }
    expect(screen.queryByText(/Invalid Date|NaN/)).not.toBeInTheDocument();
  });

  it("renders HTML-like content as plain text", () => {
    const response = fullResponse();
    response.answer = '<script>alert("sample")</script> [E1]';
    render(<AnswerResult response={response} />);
    const answer = within(
      screen.getByRole("region", { name: "Answer" }),
    ).getByText(/alert/);
    expect(answer).toBeVisible();
    expect(answer.textContent).toBe(response.answer);
    expect(document.querySelector("script")).toBeNull();
  });

  it("turns a resolvable citation into a control and leaves the rest as text", () => {
    const response = fullResponse();
    response.answer = "Supported. [E1] Unsupported. [E9]";
    render(<AnswerResult response={response} />);
    const answer = within(screen.getByRole("region", { name: "Answer" }));
    expect(answer.getByText(/Supported/).textContent).toBe(response.answer);
    expect(
      answer.getByRole("button", { name: "Show the review cited as E1" }),
    ).toBeVisible();
    expect(
      answer.queryByRole("button", { name: /cited as E9/ }),
    ).not.toBeInTheDocument();
    expect(answer.getByText(/\[E9\]/)).toBeVisible();
  });

  it("reveals and marks the cited review when a collapsed citation is used", () => {
    render(<AnswerResult response={fullResponse()} />);
    expect(
      screen.queryByRole("article", { name: "Evidence E4" }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      within(screen.getByRole("region", { name: "Answer" })).getByRole(
        "button",
        { name: "Show the review cited as E4" },
      ),
    );
    const card = screen.getByRole("article", { name: "Evidence E4" });
    expect(card).toHaveAttribute("aria-current", "true");
    expect(card).toHaveFocus();
    expect(
      screen.getByRole("button", { name: "Show less evidence" }),
    ).toBeVisible();
    expect(
      screen.getByRole("article", { name: "Evidence E1" }),
    ).not.toHaveAttribute("aria-current");
  });

  it("drops a selection that collapsing would hide", () => {
    render(<AnswerResult response={fullResponse()} />);
    fireEvent.click(
      within(screen.getByRole("region", { name: "Findings" })).getByRole(
        "button",
        { name: "Show the review cited as E4" },
      ),
    );
    expect(screen.getByRole("article", { name: "Evidence E4" })).toHaveAttribute(
      "aria-current",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Show less evidence" }));
    expect(
      screen.queryByRole("article", { name: "Evidence E4" }),
    ).not.toBeInTheDocument();
    for (const citation of screen.getAllByRole("button", {
      name: /cited as E/,
    })) {
      expect(citation).not.toHaveAttribute("aria-current");
    }
  });

  it("points to the limitations without restating or replacing them", () => {
    const response = fullResponse();
    render(<AnswerResult response={response} />);
    const answer = within(screen.getByRole("region", { name: "Answer" }));
    expect(
      answer.getByText("1 limitation and 2 warnings apply to this answer."),
    ).toBeVisible();
    expect(answer.getByRole("link", { name: "Read them" })).toHaveAttribute(
      "href",
      "#limitations",
    );
    expect(answer.queryByText(response.limitations[0])).not.toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: "Limitations" })).getByText(
        response.limitations[0],
      ),
    ).toBeVisible();
  });

  it("omits the limitation pointer when the answer carries no caveats", () => {
    render(
      <AnswerResult
        response={{ ...fullResponse(), limitations: [], warnings: [] }}
      />,
    );
    expect(screen.queryByRole("link", { name: /Read/ })).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Limitations" }),
    ).not.toBeInTheDocument();
  });
});
