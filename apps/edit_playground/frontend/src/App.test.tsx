import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App, { editAutoUsesXvector } from "./App";
import { studioApi } from "./api";

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => () => {}, destroy: () => {}, playPause: () => {}, pause: () => {} }),
  },
}));

const snapshot = {
  session_id: "browser",
  prompt: null,
  edit_source: null,
  revisions: [],
  latest: null,
  presets: [],
  defaults: { target_text: "Ready to speak.", edit_operations: [], enhance: false },
  ode_methods: ["euler", "midpoint", "rk4"],
};

const health = {
  frontend: "ready",
  model: "ready",
  ready: true,
  optimize: true,
  precision: "bfloat16",
  runtime_loaded: true,
  warmup: "complete" as const,
  max_generate_length: 512,
  uploads_enabled: true,
  demo_url: "https://dots-studio-dots-tts-edit-demo.static.hf.space",
  paper_url: "https://arxiv.org/abs/2608.02673",
};
const editSegment = { id: "segment-1", source_id: "segment-1", audio_url: "/source.wav", transcript: "A😀 source", duration_seconds: 2, revision_id: null, multiple_speakers: false, allow_xvector: true };
const editSnapshot = {
  ...snapshot,
  edit_source: { source_id: "edit_source", audio_url: "/source.wav", text: "A😀 source", revision_id: null, duration_seconds: 2, composition_version: "v1", use_xvector: true, segments: [editSegment], origin: { kind: "preset" as const, name: "Starter" } },
  presets: [{ name: "Starter", text: "A😀 source" }],
  defaults: { ...snapshot.defaults, edit_operations: [{ segment_id: "segment-1", id: "starter", kind: "delete" as const, start: 0, end: 1, params: {} }] },
};

describe("Studio layout", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    sessionStorage.clear();
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : snapshot,
    })));
  });

  it("previews the same Edit Auto policy used by the backend", () => {
    expect(editAutoUsesXvector([], false)).toBe(true);
    expect(editAutoUsesXvector([], true)).toBe(false);
    expect(editAutoUsesXvector([{ segment_id: "segment-1", id: "emo", kind: "emotion", start: 0, end: 1, params: { type: "happy" } }], false)).toBe(false);
    expect(editAutoUsesXvector([{ segment_id: "segment-1", id: "pitch", kind: "pitch", start: 0, end: 1, params: { semitones: 5 } }], true)).toBe(true);
  });

  it("renders when browser session storage is blocked", async () => {
    vi.stubGlobal("sessionStorage", {
      getItem: () => { throw new DOMException("blocked", "SecurityError"); },
      setItem: () => { throw new DOMException("blocked", "SecurityError"); },
    });
    render(<App />);
    expect(await screen.findByRole("heading", { name: "dots.tts.edit" })).toBeInTheDocument();
  });

  it("shows project status and the bilingual usage note", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : snapshot,
    })));
    render(<App />);

    const demoLink = await screen.findByRole("link", { name: "Demo" });
    expect(demoLink).toHaveAttribute(
      "href",
      "https://dots-studio-dots-tts-edit-demo.static.hf.space",
    );
    expect(demoLink.querySelector("svg")).not.toBeNull();
    const paperLink = screen.getByRole("link", { name: "arXiv Paper" });
    expect(paperLink).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2608.02673",
    );
    expect(paperLink.querySelector("img")).not.toBeNull();
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("please keep the transcript accurate");
    expect(note).toHaveTextContent("请确保转录准确");
  });

  it("shows and remembers dismissal of the source transcript hint", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : editSnapshot,
    })));
    render(<App />);

    const dismiss = await screen.findByRole("button", {
      name: "Dismiss Source Transcript hint",
    });
    fireEvent.click(dismiss);

    expect(
      screen.queryByRole("button", { name: "Dismiss Source Transcript hint" }),
    ).not.toBeInTheDocument();
    expect(sessionStorage.getItem("dots-source-transcript-hint-dismissed")).toBe("true");
  });

  it("fails closed before health resolves so upload controls never flash", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      if (String(input).includes("healthz")) return new Promise(() => {});
      return Promise.resolve({ ok: true, json: async () => editSnapshot });
    }));
    render(<App />);
    expect(await screen.findByText("Source Segment")).toBeInTheDocument();
    expect(screen.queryByText("Full Source Audio")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Choose audio" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Insert audio/ })).toHaveLength(2);
  });

  it("opens the preset and revision library from segment gaps when uploads are disabled", async () => {
    vi.spyOn(studioApi, "audioLibrary").mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, total_pages: 1,
    });
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz")
        ? { ...health, uploads_enabled: false, recognition_available: false }
        : editSnapshot,
    })));
    const { container } = render(<App />);

    fireEvent.click((await screen.findAllByRole("button", { name: /Insert audio/ }))[0]);
    const dialog = await screen.findByRole("dialog", { name: "Add Audio Segment" });
    expect(within(dialog).getByRole("tab", { name: "Presets" })).toBeInTheDocument();
    expect(within(dialog).getByRole("tab", { name: "Revisions" })).toBeInTheDocument();
    expect(within(dialog).queryByRole("tab", { name: "Uploads" })).not.toBeInTheDocument();
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument();
  });

  it("hides every file and recognition surface while retaining preset noise", async () => {
    vi.spyOn(studioApi, "noiseLibrary").mockResolvedValue({
      items: [], page: 1, page_size: 10, total: 0, total_pages: 1,
    });
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz")
        ? { ...health, uploads_enabled: false, recognition_available: false }
        : editSnapshot,
    })));
    const { container } = render(<App />);
    const noise = await screen.findByRole("button", { name: "Add background noise to Full Source Audio" });
    expect(screen.queryByRole("button", { name: "Choose audio" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Insert audio/ })).toHaveLength(2);
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Recognition" })).not.toBeInTheDocument();

    fireEvent.click(noise);
    const dialog = screen.getByRole("dialog", { name: "Background Noise for Full Source Audio" });
    expect(within(dialog).getByRole("tab", { name: "Presets" })).toBeInTheDocument();
    expect(within(dialog).queryByRole("tab", { name: "Uploads" })).not.toBeInTheDocument();
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument();
  });

  it("keeps voice prompts separate from edit-source examples", async () => {
    const splitSnapshot = {
      ...editSnapshot,
      prompt: {
        ...editSnapshot.edit_source,
        source_id: "prompt",
        origin: { kind: "preset" as const, name: "female_en" },
      },
      presets: [
        { name: "female_en", text: "Voice prompt" },
        { name: "text_en", text: "Edit source" },
      ],
      prompt_presets: [{ name: "female_en", text: "Voice prompt" }],
      edit_source_presets: [{ name: "text_en", text: "Edit source" }],
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : splitSnapshot,
    })));
    const { container } = render(<App />);

    await waitFor(() => expect(container.querySelector(".edit-section .preset-select select")).toHaveValue("text_en"));
    const editOptions = Array.from(
      container.querySelectorAll<HTMLOptionElement>(".edit-section .preset-select option"),
    ).map((option) => option.value);
    const promptOptions = Array.from(
      container.querySelectorAll<HTMLOptionElement>(".zero-shot-section .preset-select option"),
    ).map((option) => option.value);
    expect(editOptions).toContain("text_en");
    expect(editOptions).not.toContain("female_en");
    expect(promptOptions).toContain("female_en");
    expect(promptOptions).not.toContain("text_en");
  });

  it("keeps independent generation config behind each split button", async () => {
    render(<App />);
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    expect(screen.queryByText("Generator Settings")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Zero-shot TTS" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Speech Edit" })).toBeInTheDocument();
    const headings = screen.getAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.textContent)).toEqual(["Speech Edit", "Zero-shot TTS"]);
    expect(screen.getByRole("button", { name: "Apply Edit" })).toBeDisabled();
    expect(screen.getByDisplayValue("Ready to speak.")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "ODE Method" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Synthesize generation config" }));
    expect(screen.getByRole("combobox", { name: "ODE Method" })).toHaveValue("euler");
    expect(screen.getByRole("button", { name: "Expand history" })).toBeInTheDocument();
    expect(screen.queryByText("Language")).not.toBeInTheDocument();
  });

  it("replaces the latest-result media area with generation progress", async () => {
    vi.spyOn(studioApi, "tts").mockImplementation(async (_session, _text, _settings, onEvent) => {
      onEvent({ phase: "inference", progress: null });
      await new Promise(() => {});
    });
    const { container } = render(<App />);
    const synthesize = await waitFor(() => {
      const element = container.querySelector<HTMLButtonElement>(".zero-shot-section .primary-button");
      expect(element).toBeInTheDocument();
      return element!;
    });
    await waitFor(() => expect(synthesize).toBeEnabled());
    fireEvent.click(synthesize);
    await waitFor(() => expect(container.querySelector(".generation-progress")).toBeInTheDocument());
    expect(container).not.toHaveTextContent("Latest Result");
  });

  it("keeps Edit and Zero-shot generation settings independent", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : editSnapshot,
    })));
    const { container } = render(<App />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Apply Edit generation config" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Apply Edit generation config" }));
    const editConfig = container.querySelector(".edit-section .generation-config")!;
    const editCollapse = container.querySelector(".edit-section .generation-config-collapse")!;
    expect(editCollapse).toHaveAttribute("aria-hidden", "false");
    const editSteps = within(editConfig as HTMLElement).getByRole("slider", { name: /Steps/ });
    fireEvent.change(editSteps, { target: { value: "11" } });
    expect(editSteps).toHaveValue("11");

    fireEvent.click(screen.getByRole("button", { name: "Synthesize generation config" }));
    expect(editCollapse).toHaveAttribute("aria-hidden", "true");
    const ttsConfig = container.querySelector(".zero-shot-section .generation-config")!;
    expect(within(ttsConfig as HTMLElement).getByRole("slider", { name: /Steps/ })).toHaveValue("32");
  });

  it("switches the session model without resetting the edit workspace", async () => {
    const models = [
      { id: "alternate", label: "Alternate", path: "/models/alternate", status: "ready" as const, error: null, runtime_loaded: true },
      { id: "secondary", label: "Secondary", path: "/models/secondary", status: "ready" as const, error: null, runtime_loaded: true },
    ];
    const initial = { ...editSnapshot, selected_model_id: "alternate", models };
    const next = { ...initial, selected_model_id: "secondary" };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz")
        ? { ...health, default_model_id: "alternate", runtime_loaded_count: 2, models }
        : initial,
    })));
    const selectModel = vi.spyOn(studioApi, "selectModel").mockResolvedValue(next);
    const { container } = render(<App />);
    const selector = await screen.findByRole("combobox", { name: "Model" });
    expect(selector).toHaveValue("alternate");
    expect(container.querySelector(".operation-chip")).toHaveTextContent("Delete");

    fireEvent.change(selector, { target: { value: "secondary" } });

    await waitFor(() => expect(selectModel).toHaveBeenCalledWith(expect.any(String), "secondary"));
    expect(selector).toHaveValue("secondary");
    expect(container.querySelector(".operation-chip")).toHaveTextContent("Delete");
  });

  it("allows Enhance as the only operation and sends it through preview and generation", async () => {
    const noOperations = {
      ...editSnapshot,
      defaults: { ...editSnapshot.defaults, edit_operations: [] },
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : noOperations,
    })));
    const compile = vi.spyOn(studioApi, "compile").mockResolvedValue({
      instruction: "<enhance>A😀 source</enhance>",
      target_text: "A😀 source",
    });
    const edit = vi.spyOn(studioApi, "edit").mockResolvedValue();
    render(<App />);
    const enhance = await screen.findByRole("button", { name: "Enhance" });
    const apply = screen.getByRole("button", { name: "Apply Edit" });
    expect(enhance.closest(".source-global-controls")).toHaveTextContent(
      "1 segment · 2.0s / 40s",
    );
    expect(apply).toBeDisabled();

    fireEvent.click(enhance);
    expect(enhance).toHaveAttribute("aria-pressed", "true");
    expect(apply).toBeEnabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Preview" }));
    await waitFor(() => expect(compile).toHaveBeenCalledWith(
      expect.any(String),
      "v1",
      [],
      true,
    ));

    fireEvent.click(apply);
    await waitFor(() => expect(edit).toHaveBeenCalledOnce());
    expect(edit.mock.calls[0][3]).toMatchObject({ enhance: true });
  });

  it("defaults Edit speaker guidance to Auto at scale 1.5", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : editSnapshot,
    })));
    const edit = vi.spyOn(studioApi, "edit").mockResolvedValue();
    const { container } = render(<App />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Apply Edit" })).toBeEnabled());
    expect(edit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Apply Edit" }));
    await waitFor(() => expect(edit).toHaveBeenCalledOnce());
    expect(edit.mock.calls[0][3]).toMatchObject({
      use_xvector: "auto",
      speaker_scale: 1.5,
    });
    expect(container.querySelector(".edit-section .generation-config-collapse")).toHaveAttribute("aria-hidden", "true");
  });

  it("edits Source Transcript in place and preserves operations when unchanged", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : editSnapshot,
    })));
    const { container } = render(<App />);
    const view = within(container);
    await waitFor(() => expect(view.getByRole("button", { name: "Edit Source Transcript" })).toBeEnabled());
    await waitFor(() => expect(container.querySelector(".operation-chip")).toBeInTheDocument());
    fireEvent.click(view.getByRole("button", { name: "Edit Source Transcript" }));
    expect(view.getByRole("textbox", { name: "Source Transcript" })).toHaveValue("A😀 source");
    expect(container.querySelector(".operation-chip")).not.toBeInTheDocument();
    expect(container.querySelector<HTMLButtonElement>(".edit-section .primary-button")).toBeDisabled();
    fireEvent.click(view.getByRole("button", { name: /Save/ }));
    await waitFor(() => expect(container.querySelector(".operation-chip")).toBeInTheDocument());
  });

  it("confirms a changed Source Transcript and clears stale operations", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : editSnapshot,
    })));
    const next = { ...editSnapshot, edit_source: { ...editSnapshot.edit_source, text: "Changed source", composition_version: "v2", segments: [{ ...editSegment, transcript: "Changed source" }] }, defaults: { ...editSnapshot.defaults, edit_operations: [] } };
    vi.spyOn(studioApi, "updateSegmentTranscript").mockResolvedValue(next);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const { container } = render(<App />);
    const view = within(container);
    await waitFor(() => expect(view.getByRole("button", { name: "Edit Source Transcript" })).toBeEnabled());
    fireEvent.click(view.getByRole("button", { name: "Edit Source Transcript" }));
    fireEvent.change(view.getByRole("textbox", { name: "Source Transcript" }), { target: { value: "Changed source" } });
    fireEvent.click(view.getByRole("button", { name: /Save/ }));
    await waitFor(() => expect(studioApi.updateSegmentTranscript).toHaveBeenCalledWith(expect.any(String), "segment-1", "Changed source", "v1"));
    expect(confirm).toHaveBeenCalledWith("Changing this Source Transcript will remove its edit operations.");
    await waitFor(() => expect(container.querySelector(".operation-chip")).not.toBeInTheDocument());
  });

  it("keeps an unsaved Source Transcript draft when a source switch is cancelled", async () => {
    const guardedSnapshot = { ...editSnapshot, presets: [...editSnapshot.presets, { name: "Second", text: "Second source" }] };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : guardedSnapshot,
    })));
    const preset = vi.spyOn(studioApi, "preset");
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const { container } = render(<App />);
    const view = within(container);
    await waitFor(() => expect(view.getByRole("button", { name: "Edit Source Transcript" })).toBeEnabled());
    fireEvent.click(view.getByRole("button", { name: "Edit Source Transcript" }));
    const editor = view.getByRole("textbox", { name: "Source Transcript" });
    fireEvent.change(editor, { target: { value: "Unsaved draft" } });
    const editPreset = container.querySelector<HTMLSelectElement>(".edit-section .preset-select select")!;
    fireEvent.change(editPreset, { target: { value: "Second" } });
    expect(preset).not.toHaveBeenCalled();
    expect(editor).toHaveValue("Unsaved draft");
  });

  it("loads structured starter operations whenever the Edit Source preset changes", async () => {
    const initial = { ...editSnapshot, presets: [...editSnapshot.presets, { name: "Second", text: "Second source" }] };
    const next = {
      ...initial,
      edit_source: { ...editSnapshot.edit_source, audio_url: "/second.wav", text: "Second source", origin: { kind: "preset" as const, name: "Second" } },
      defaults: { ...initial.defaults, edit_operations: [{ segment_id: "segment-1", id: "second-rate", kind: "rate" as const, start: 0, end: 6, params: { factor: 0.55 } }] },
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : initial,
    })));
    vi.spyOn(studioApi, "preset").mockResolvedValue(next);
    const { container } = render(<App />);
    await waitFor(() => expect(container.querySelector(".operation-chip")).toHaveTextContent("Delete"));

    fireEvent.change(container.querySelector<HTMLSelectElement>(".edit-section .preset-select select")!, { target: { value: "Second" } });

    await waitFor(() => expect(container.querySelector(".operation-chip")).toHaveTextContent("Rate"));
    expect(container.querySelector(".operation-chip")).not.toHaveTextContent("Delete");
  });

  it("renders one transcript canvas per segment and orders the shared operation overview by source position", async () => {
    const second = { ...editSegment, id: "segment-2", source_id: "segment-2", audio_url: "/second.wav", transcript: "second" };
    const multi = {
      ...editSnapshot,
      edit_source: {
        ...editSnapshot.edit_source,
        audio_url: "/full.wav",
        text: "A😀 source second",
        duration_seconds: 4,
        composition_version: "multi-v1",
        use_xvector: false,
        segments: [editSegment, second],
      },
      defaults: {
        ...editSnapshot.defaults,
        edit_operations: [
          { segment_id: "segment-2", id: "late", kind: "replace" as const, start: 0, end: 3, params: { target: "third" } },
          { segment_id: "segment-1", id: "middle", kind: "pause" as const, start: 3, end: 3, params: { act: "ins", level: 2 } },
          { segment_id: "segment-1", id: "early", kind: "delete" as const, start: 0, end: 1, params: {} },
        ],
      },
    };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve({
      ok: true,
      json: async () => String(input).includes("healthz") ? health : multi,
    })));
    const { container } = render(<App />);
    await waitFor(() => expect(container.querySelectorAll(".segment-transcript .edit-canvas")).toHaveLength(2));
    const chips = Array.from(container.querySelectorAll<HTMLElement>(".operation-chip"));
    expect(chips.map((chip) => chip.textContent)).toEqual([
      expect.stringContaining("Delete"),
      expect.stringContaining("Pause"),
      expect.stringContaining("Replace"),
    ]);
    fireEvent.mouseEnter(chips[0]);
    expect(container.querySelector('[data-operation-id="early"].source-char')).toHaveClass("operation-mark--linked");
    fireEvent.mouseLeave(chips[0]);
    expect(container.querySelector('[data-operation-id="early"].source-char')).not.toHaveClass("operation-mark--linked");
    fireEvent.mouseMove(container.querySelector('[data-operation-id="early"].source-char')!);
    expect(chips[0]).toHaveClass("operation-chip--linked");
    fireEvent.click(chips[2]);
    await waitFor(() => expect(screen.getByText("Edit operation")).toBeInTheDocument());
  });
});
