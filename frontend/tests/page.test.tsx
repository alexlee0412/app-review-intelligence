import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Home from "../app/page";
import { errorMessages } from "../lib/ask";
import { fullResponse } from "./fixtures";

const internalDetail =
  "INTERNAL_FAILURE: private-provider private-model connection details";

beforeEach(() =>
  vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://reviews.example.test/"),
);

async function enterQuestion() {
  const user = userEvent.setup();
  await user.type(
    screen.getByRole("textbox", { name: "Your question" }),
    "  What needs improvement?  ",
  );
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return user;
}

describe("question workflow", () => {
  it("posts only the question to the configured endpoint and renders a validated result", async () => {
    const fetcher = vi.fn().mockResolvedValue(Response.json(fullResponse()));
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    await enterQuestion();
    expect(
      await screen.findByRole("heading", { name: "Answer" }),
    ).toBeVisible();
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(
      new URL("https://reviews.example.test/api/v1/reviews/ask"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "What needs improvement?" }),
        cache: "no-store",
        credentials: "omit",
      },
    );
    expect(screen.getByRole("status")).toHaveTextContent("Analysis ready");
  });

  it("populates an example without sending a request and focuses the input", async () => {
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    const example = "What problems are users reporting about EPIK?";
    await userEvent.click(screen.getByRole("button", { name: example }));
    expect(screen.getByRole("textbox")).toHaveValue(example);
    expect(screen.getByRole("textbox")).toHaveFocus();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("disables Ask during loading and guards duplicate Enter and form submissions", async () => {
    let complete!: (response: Response) => void;
    const fetcher = vi.fn().mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          complete = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    const user = await enterQuestion();
    expect(screen.getByRole("button", { name: "Analyzing…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Analyzing app reviews…",
    );
    const input = screen.getByRole("textbox");
    expect(input).toBeDisabled();
    await user.keyboard("{Enter}{Enter}");
    fireEvent.submit(input.closest("form")!);
    fireEvent.submit(input.closest("form")!);
    expect(fetcher).toHaveBeenCalledTimes(1);
    await act(async () => complete(Response.json(fullResponse())));
    expect(await screen.findByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("prevents blank and whitespace-only questions", async () => {
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    await userEvent.type(screen.getByRole("textbox"), "   ");
    fireEvent.submit(screen.getByRole("textbox").closest("form")!);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("clears an old result when another question starts and resets evidence expansion", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(Response.json(fullResponse()));
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    await enterQuestion();
    await screen.findByRole("heading", { name: "Answer" });
    await userEvent.click(
      screen.getByRole("button", { name: "Show more evidence" }),
    );
    let complete!: (response: Response) => void;
    fetcher.mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          complete = resolve;
        }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(
      screen.queryByRole("heading", { name: "Answer" }),
    ).not.toBeInTheDocument();
    await act(async () => complete(Response.json(fullResponse())));
    expect(
      await screen.findByRole("button", { name: "Show more evidence" }),
    ).toHaveAttribute("aria-expanded", "false");
  });
});

describe("friendly errors and retry", () => {
  it.each([
    [400, errorMessages.question],
    [422, errorMessages.question],
    [503, errorMessages.unavailable],
    [500, errorMessages.other],
    [401, errorMessages.other],
  ])(
    "maps HTTP %s without exposing response details",
    async (status, message) => {
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValue(
            Response.json({ detail: internalDetail }, { status }),
          ),
      );
      render(<Home />);
      await enterQuestion();
      expect(await screen.findByRole("alert")).toHaveTextContent(message);
      expect(document.body).not.toHaveTextContent(internalDetail);
      expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
    },
  );

  it("handles network failure and lets the same question succeed on retry", async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new Error(internalDetail))
      .mockResolvedValueOnce(Response.json(fullResponse()));
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    await enterQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      errorMessages.network,
    );
    expect(document.body).not.toHaveTextContent(internalDetail);
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(
      await screen.findByRole("heading", { name: "Answer" }),
    ).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["invalid JSON", () => new Response(internalDetail)],
    ["missing fields", () => Response.json({ answer: internalDetail })],
    [
      "invalid nested evidence",
      () =>
        Response.json({
          ...fullResponse(),
          evidence: [{ excerpt: internalDetail }],
        }),
    ],
    [
      "invalid metrics",
      () =>
        Response.json({
          ...fullResponse(),
          metrics: { totals: { total_reviews: internalDetail } },
        }),
    ],
    ["invalid trace", () => Response.json({ ...fullResponse(), trace: null })],
    [
      "a finding without evidence ids",
      () =>
        Response.json({
          ...fullResponse(),
          findings: [
            { claim: internalDetail, kind: "computed", evidence_ids: [] },
          ],
        }),
    ],
  ])("handles a successful HTTP response with %s", async (_label, response) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response()));
    render(<Home />);
    await enterQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      errorMessages.response,
    );
    expect(document.body).not.toHaveTextContent(internalDetail);
    expect(
      screen.queryByRole("heading", { name: "Answer" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
  });

  it.each([
    "",
    "invalid-url",
    "file:///invalid",
    "https://reviews.example.test/?bad=path",
  ])("handles invalid configuration: %s", async (base) => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", base);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    render(<Home />);
    await enterQuestion();
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        errorMessages.configuration,
      ),
    );
    expect(fetcher).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
  });
});
