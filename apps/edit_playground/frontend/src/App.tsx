import { useEffect, useMemo, useRef, useState } from "react";
import { Headphones, LoaderCircle, Sparkles, WandSparkles, X } from "lucide-react";
import arxivLogo from "./assets/arxiv-logo.svg";
import { studioApi } from "./api";
import { createId, readSessionValue, writeSessionValue } from "./browser";
import type { AudioCardData } from "./components/AudioCard";
import { EditCanvas, operationStyle, TranscriptHint } from "./components/EditCanvas";
import { AudioLibraryDialog, type AudioLibraryIntent } from "./components/AudioLibraryDialog";
import { GenerationTransport } from "./components/GenerationTransport";
import { GeneratorSettingsPanel, SplitGenerationButton } from "./components/GeneratorSettings";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { ReferencePanel } from "./components/ReferencePanel";
import { SourceAudioTimeline } from "./components/SourceAudioTimeline";
import { OperationList } from "./components/OperationList";
import { NoiseLibraryDialog } from "./components/NoiseLibraryDialog";
import { OnboardingTour } from "./components/OnboardingTour";
import type { AudioLibraryItem, CompiledEdit, Destination, EditGenerationSettings, EditOperation, GenerationEvent, GenerationSettings, HealthStatus, SessionSnapshot } from "./types";

const DEFAULT_SETTINGS: GenerationSettings = { ode_method: "euler", num_steps: 32, guidance_scale: 1, speaker_scale: 1.5, use_xvector: true, seed: 20260414 };
const DEFAULT_EDIT_SETTINGS: EditGenerationSettings = {
  ...DEFAULT_SETTINGS,
  use_xvector: "auto",
};
const HINT_KEY = "dots-source-transcript-hint-dismissed";
const POINT_KINDS = new Set(["insert", "pause"]);

function sessionId() {
  const existing = readSessionValue("dots-studio-session");
  if (existing) return existing;
  const value = createId();
  writeSessionValue("dots-studio-session", value);
  return value;
}

function latestData(snapshot: SessionSnapshot | null): AudioCardData | null {
  const latest = snapshot?.latest;
  return latest ? { sourceId: latest.id, audioUrl: latest.audio_url, text: latest.text } : null;
}

function orderedOperations(snapshot: SessionSnapshot | null, operations: EditOperation[]) {
  const segmentOrder = new Map(
    (snapshot?.edit_source?.segments || []).map((segment, index) => [segment.id, index]),
  );
  return [...operations].sort((left, right) =>
    (segmentOrder.get(left.segment_id) ?? Number.MAX_SAFE_INTEGER) - (segmentOrder.get(right.segment_id) ?? Number.MAX_SAFE_INTEGER)
    || left.start - right.start
    || Number(!POINT_KINDS.has(left.kind)) - Number(!POINT_KINDS.has(right.kind))
    || left.end - right.end
    || left.id.localeCompare(right.id),
  );
}

export function editAutoUsesXvector(operations: EditOperation[], enhance: boolean) {
  if (!operations.length && !enhance) return true;
  return operations.some((operation) => operation.kind !== "emotion");
}

export default function App() {
  const session = useMemo(sessionId, []);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [editSettings, setEditSettings] = useState(DEFAULT_EDIT_SETTINGS);
  const [ttsSettings, setTtsSettings] = useState(DEFAULT_SETTINGS);
  const [editConfigOpen, setEditConfigOpen] = useState(false);
  const [ttsConfigOpen, setTtsConfigOpen] = useState(false);
  const [targetTranscript, setTargetTranscript] = useState("");
  const [operations, setOperations] = useState<EditOperation[]>([]);
  const [enhance, setEnhance] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [preview, setPreview] = useState<CompiledEdit | null>(null);
  const [status, setStatus] = useState<GenerationEvent>({ phase: "idle", progress: 0 });
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [editingSegmentId, setEditingSegmentId] = useState<string | null>(null);
  const [transcriptDraft, setTranscriptDraft] = useState("");
  const [requestedEditId, setRequestedEditId] = useState<string | null>(null);
  const [hoveredOperationId, setHoveredOperationId] = useState<string | null>(null);
  const [hintVisible, setHintVisible] = useState(() => readSessionValue(HINT_KEY) !== "true");
  const [error, setError] = useState("");
  const [routedTo, setRoutedTo] = useState<Destination | null>(null);
  const [libraryIntent, setLibraryIntent] = useState<AudioLibraryIntent | null>(null);
  const [noiseDialogOpen, setNoiseDialogOpen] = useState(false);
  const promptTimer = useRef<number | null>(null);
  const editPair = useRef<HTMLDivElement>(null);
  const promptPair = useRef<HTMLDivElement>(null);

  useEffect(() => {
    studioApi.snapshot(session).then((next) => {
      setSnapshot(next);
      setTargetTranscript(next.defaults.target_text);
      setOperations(next.defaults.edit_operations);
      setEnhance(next.defaults.enhance);
    }).catch((reason: unknown) => setError(errorMessage(reason)));
  }, [session]);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const next = await studioApi.health();
        if (active) {
          setHealth(next);
          if (next.model === "error" && next.model_error) setError(`Model startup failed: ${next.model_error}`);
        }
        if (active && next.model === "warming") window.setTimeout(poll, 1200);
      } catch {
        if (active) window.setTimeout(poll, 1600);
      }
    };
    void poll();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const version = snapshot?.edit_source?.composition_version;
    if (!previewOpen || editingSegmentId || !version || (!operations.length && !enhance)) return;
    const handle = window.setTimeout(() => studioApi.compile(session, version, operations, enhance).then(setPreview).catch((reason: unknown) => {
      setPreview(null);
      setError(errorMessage(reason));
    }), 100);
    return () => window.clearTimeout(handle);
  }, [previewOpen, editingSegmentId, operations, enhance, session, snapshot?.edit_source?.composition_version]);

  useEffect(() => () => {
    if (promptTimer.current) window.clearTimeout(promptTimer.current);
  }, []);

  const applySnapshot = (next: SessionSnapshot, resetSource = false) => {
    setSnapshot(next);
    if (resetSource) {
      setOperations(next.defaults.edit_operations || []);
      setPreview(null);
      setEditingSegmentId(null);
      setTranscriptDraft("");
      setRequestedEditId(null);
      setEnhance(next.defaults.enhance || false);
    }
    setError("");
  };

  const cancelPromptUpdate = () => {
    if (promptTimer.current) window.clearTimeout(promptTimer.current);
    promptTimer.current = null;
  };

  const discardDraftForSourceChange = () => {
    if (!editingSegmentId) return true;
    const original = snapshot?.edit_source?.segments.find((segment) => segment.id === editingSegmentId)?.transcript || "";
    if (transcriptDraft === original || window.confirm("Discard the unsaved Source Transcript changes?")) {
      setEditingSegmentId(null);
      setTranscriptDraft("");
      return true;
    }
    return false;
  };

  const route = async (source: string, destination: Destination) => {
    if (destination === "edit_source" && !discardDraftForSourceChange()) return;
    try {
      if (destination === "prompt") cancelPromptUpdate();
      applySnapshot(await studioApi.route(session, source, destination), destination === "edit_source");
      setHistoryExpanded(false);
      setRoutedTo(destination);
      window.setTimeout(() => {
        (destination === "prompt" ? promptPair : editPair).current?.scrollIntoView({ behavior: "smooth", block: "center" });
        window.setTimeout(() => setRoutedTo(null), 900);
      }, 0);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const requestUpload = (destination: Destination, file?: File) => {
    if (health?.uploads_enabled !== true) return;
    if (destination === "edit_source" && !discardDraftForSourceChange()) return;
    setLibraryIntent({
      destination,
      action: "bind",
      title: destination === "prompt" ? "Choose Voice Prompt" : "Choose Edit Source",
      initialFile: file,
      currentDuration: snapshot?.edit_source?.duration_seconds || 0,
      currentSegmentCount: snapshot?.edit_source?.segments.length || 0,
    });
  };

  const choosePreset = async (destination: Destination, value: string) => {
    if (destination === "edit_source" && !discardDraftForSourceChange()) return;
    try {
      if (destination === "prompt") cancelPromptUpdate();
      applySnapshot(value === "__random__" ? await studioApi.clearReference(session, destination) : await studioApi.preset(session, value, destination), destination === "edit_source");
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const updatePromptTranscript = (text: string) => {
    setSnapshot((current) => current?.prompt ? { ...current, prompt: { ...current.prompt, text } } : current);
    if (promptTimer.current) window.clearTimeout(promptTimer.current);
    promptTimer.current = window.setTimeout(() => studioApi.updateText(session, "prompt", text).catch((reason: unknown) => setError(errorMessage(reason))), 250);
  };

  const onGenerationEvent = (event: GenerationEvent) => {
    setStatus(event);
    if (event.snapshot) applySnapshot(event.snapshot);
    if (event.phase === "error") setError(event.message || "Generation failed.");
  };

  const generateTts = async () => {
    if (!targetTranscript.trim()) return setError("Enter the transcript to synthesize.");
    if (snapshot?.prompt && !snapshot.prompt.text.trim()) return setError("Prompt Audio and Prompt Transcript must be provided together.");
    setError("");
    setTtsConfigOpen(false);
    setStatus({ phase: "queued", progress: 0 });
    try {
      if (snapshot?.prompt) {
        cancelPromptUpdate();
        applySnapshot(await studioApi.updateText(session, "prompt", snapshot.prompt.text));
      }
      await studioApi.tts(session, targetTranscript, {
        ...ttsSettings,
        model_id: snapshot?.selected_model_id || "",
        use_xvector: ttsSettings.use_xvector && (snapshot?.prompt?.use_xvector ?? false),
      }, onGenerationEvent);
    } catch (reason) {
      setStatus({ phase: "error", progress: null });
      setError(errorMessage(reason));
    }
  };

  const generateEdit = async () => {
    if (!snapshot?.edit_source) return setError("Choose or upload Source Audio first.");
    if (snapshot.edit_source.segments.some((segment) => !segment.transcript.trim())) return setError("Enter the transcript paired with each Source Audio segment.");
    if (!operations.length && !enhance) return setError("Add at least one edit operation.");
    setError("");
    setEditConfigOpen(false);
    setStatus({ phase: "queued", progress: 0 });
    try {
      await studioApi.edit(session, snapshot.edit_source.composition_version, operations, {
        ...editSettings,
        model_id: snapshot.selected_model_id || "",
        enhance,
        use_xvector: editSettings.use_xvector,
      }, onGenerationEvent);
    } catch (reason) {
      setStatus({ phase: "error", progress: null });
      setError(errorMessage(reason));
    }
  };

  const saveSourceTranscript = async (segmentId: string) => {
    const source = snapshot?.edit_source;
    const segment = source?.segments.find((item) => item.id === segmentId);
    if (!source || !segment) return;
    if (transcriptDraft === segment.transcript) {
      setEditingSegmentId(null);
      return;
    }
    const segmentOperations = operations.filter((operation) => operation.segment_id === segmentId);
    if (segmentOperations.length && !window.confirm("Changing this Source Transcript will remove its edit operations.")) return;
    try {
      const next = await studioApi.updateSegmentTranscript(session, segmentId, transcriptDraft, source.composition_version);
      setOperations((current) => current.filter((operation) => operation.segment_id !== segmentId));
      setPreview(null);
      setEditingSegmentId(null);
      setTranscriptDraft("");
      applySnapshot(next);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const applySegmentSnapshot = (next: SessionSnapshot) => {
    const segmentIds = new Set(next.edit_source?.segments.map((segment) => segment.id) || []);
    setOperations((current) => current.filter((operation) => segmentIds.has(operation.segment_id)));
    setPreview(null);
    applySnapshot(next);
  };

  const insertSegment = async (sourceId: string, index: number) => {
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source; if (!source) return;
    try { applySegmentSnapshot(await studioApi.insertSegment(session, sourceId, index, source.composition_version)); } catch (reason) { setError(errorMessage(reason)); }
  };

  const moveSegment = (segmentId: string, index: number) => {
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source; if (!source) return;
    void studioApi.mutateSegment(session, segmentId, { action: "move", to_index: index, expected_version: source.composition_version }).then(applySegmentSnapshot).catch((reason: unknown) => setError(errorMessage(reason)));
  };

  const replaceSegment = (segmentId: string, sourceId: string) => {
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source; if (!source) return;
    void studioApi.mutateSegment(session, segmentId, { action: "replace", source_id: sourceId, expected_version: source.composition_version }).then(applySegmentSnapshot).catch((reason: unknown) => setError(errorMessage(reason)));
  };

  const removeSegment = (segmentId: string) => {
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source; if (!source) return;
    void studioApi.deleteSegment(session, segmentId, source.composition_version).then(applySegmentSnapshot).catch((reason: unknown) => setError(errorMessage(reason)));
  };

  const uploadAt = (index: number, file?: File) => {
    if (file && health?.uploads_enabled !== true) return;
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source;
    if (!source) return;
    setLibraryIntent({ destination: "edit_source", action: "insert", index, expected_version: source.composition_version, title: "Add Audio Segment", initialFile: health?.uploads_enabled === true ? file : undefined, currentDuration: source.duration_seconds, currentSegmentCount: source.segments.length });
  };

  const uploadReplace = (segmentId: string, file?: File) => {
    if (health?.uploads_enabled !== true) return;
    if (!discardDraftForSourceChange()) return;
    const source = snapshot?.edit_source;
    const segment = source?.segments.find((item) => item.id === segmentId);
    if (!source || !segment) return;
    setLibraryIntent({ destination: "edit_source", action: "replace", replace_segment_id: segmentId, expected_version: source.composition_version, title: "Replace Audio Segment", initialFile: file, currentDuration: source.duration_seconds, currentSegmentCount: source.segments.length, replacedDuration: segment.duration_seconds });
  };

  const applyLibrarySelection = (next: SessionSnapshot, _item: AudioLibraryItem) => {
    const intent = libraryIntent;
    if (!intent) return;
    if (intent.destination === "prompt") cancelPromptUpdate();
    if (intent.action === "bind") applySnapshot(next, intent.destination === "edit_source");
    else applySegmentSnapshot(next);
    setLibraryIntent(null);
  };

  const sortedOperations = useMemo(
    () => orderedOperations(snapshot, operations),
    [operations, snapshot?.edit_source?.composition_version],
  );
  const operationColors = useMemo(
    () => Object.fromEntries(sortedOperations.map((operation) => [operation.id, operationStyle(operation.kind).color])),
    [sortedOperations],
  );
  const segmentNumbers = useMemo(
    () => Object.fromEntries((snapshot?.edit_source?.segments || []).map((segment, index) => [segment.id, index + 1])),
    [snapshot?.edit_source?.composition_version],
  );
  const updateSegmentOperations = (segmentId: string, next: EditOperation[]) => {
    setOperations((current) => [
      ...current.filter((operation) => operation.segment_id !== segmentId),
      ...next,
    ]);
  };

  const selectModel = async (modelId: string) => {
    try {
      applySnapshot(await studioApi.selectModel(session, modelId));
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const dismissHint = () => {
    writeSessionValue(HINT_KEY, "true");
    setHintVisible(false);
  };

  const busy = ["queued", "preparing", "inference", "saving"].includes(status.phase);
  const modelOptions = health?.models?.length ? health.models : snapshot?.models || [];
  const selectedModel = modelOptions.find((model) => model.id === snapshot?.selected_model_id);
  const modelReady = selectedModel?.status === "ready"
    || (health?.warmup === "skipped" && selectedModel?.status === "pending")
    || (!modelOptions.length && health?.model === "ready");
  const promptChoice = snapshot?.prompt?.origin?.kind === "preset" ? snapshot.prompt.origin.name || "" : snapshot?.prompt ? "__custom__" : "__random__";
  const editChoice = snapshot?.edit_source?.origin?.kind === "preset" ? snapshot.edit_source.origin.name || "" : "__custom__";
  const promptPresets = snapshot?.prompt_presets || snapshot?.presets || [];
  const editSourcePresets = snapshot?.edit_source_presets || snapshot?.presets || [];
  // Capability discovery is fail-closed: controls must not flash before health resolves.
  const uploadsEnabled = health?.uploads_enabled === true;

  return <div className="app-shell">
    <main className="studio-main">
      <header className="app-header"><div className="app-title"><h1>dots.tts.edit</h1><nav aria-label="Project links">{health?.demo_url && <a className="project-link" href={health.demo_url} target="_blank" rel="noreferrer"><Headphones size={17} aria-hidden="true" /><span>Demo</span></a>}{health?.paper_url && <a className="project-link project-link--arxiv" href={health.paper_url} target="_blank" rel="noreferrer" aria-label="arXiv Paper"><img src={arxivLogo} alt="" /></a>}</nav></div>
        <label className="model-selector"><span>Model</span><select aria-label="Model" disabled={busy || !snapshot} value={snapshot?.selected_model_id || ""} onChange={(event) => void selectModel(event.target.value)}>{modelOptions.map((model) => <option key={model.id} value={model.id} disabled={model.status === "error"} title={model.path}>{model.label}{model.status === "ready" ? "" : ` · ${model.status}`}</option>)}</select></label>
        <div className={`model-status ${busy || !modelReady ? "model-status--busy" : ""}`}>{busy || !modelReady ? <LoaderCircle size={13} className="spin" /> : <span />}{busy ? "Generating" : selectedModel?.status === "error" ? "Model error" : modelReady ? "Ready" : "Warming up"}</div>
      </header>

      <div className="usage-note" role="note">
        <span className="usage-note__icon" aria-hidden="true">💡</span>
        <div className="usage-note__copy">
          <p>For better results, please keep the transcript accurate.</p>
          <p lang="zh">为获得更好的效果，请确保转录准确。</p>
        </div>
      </div>
      <div data-tour="latest-results"><GenerationTransport busy={busy} status={status} title={snapshot?.latest?.metadata.kind === "edit" ? "Latest Edit" : "Latest Result"} data={latestData(snapshot)} onRoute={route} /></div>
      {error && <div className="error-banner"><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError("")}><X size={14} /></button></div>}

      <section className="workstation-section edit-section"><header className="section-header"><div><WandSparkles size={17} /><h2>Speech Edit</h2></div></header>
        <div className="edit-workspace"><ReferencePanel panelRef={editPair} title="Edit Source" destination="edit_source" reference={snapshot?.edit_source || null} presetValue={editChoice} presets={editSourcePresets} routed={routedTo === "edit_source"} onPresetChange={(value) => void choosePreset("edit_source", value)} onRoute={route} onUpload={(file) => requestUpload("edit_source", file)} uploadsEnabled={uploadsEnabled} audioContent={snapshot?.edit_source ? <SourceAudioTimeline reference={snapshot.edit_source} onRoute={route} onInsert={(source, index) => void insertSegment(source, index)} onMove={moveSegment} onReplace={replaceSegment} onRemove={removeSegment} onUploadWhole={(file) => requestUpload("edit_source", file)} onUploadAt={uploadAt} onUploadReplace={uploadReplace} uploadsEnabled={uploadsEnabled} onNoise={() => setNoiseDialogOpen(true)} enhance={enhance} onEnhanceToggle={() => setEnhance((value) => !value)} globalHint={hintVisible && !editingSegmentId ? <TranscriptHint onDismiss={dismissHint} /> : undefined} renderTranscript={(segmentId, index) => {
          const segment = snapshot.edit_source!.segments[index];
          const localOperations = sortedOperations.filter((operation) => operation.segment_id === segmentId);
          const editing = editingSegmentId === segmentId;
          return <EditCanvas segmentId={segmentId} text={segment.transcript} operations={localOperations} operationColors={operationColors} onOperationsChange={(next) => updateSegmentOperations(segmentId, next)} editing={editing} draft={editing ? transcriptDraft : ""} hintVisible={false} onBeginEdit={() => { setTranscriptDraft(segment.transcript); setEditingSegmentId(segmentId); }} onDraftChange={setTranscriptDraft} onSaveEdit={() => void saveSourceTranscript(segmentId)} onCancelEdit={() => { setTranscriptDraft(""); setEditingSegmentId(null); }} onDismissHint={dismissHint} requestedEditId={requestedEditId} onRequestedEditHandled={() => setRequestedEditId(null)} hoveredOperationId={hoveredOperationId} onHoverOperation={setHoveredOperationId} disabled={!segment.transcript.trim() || Boolean(editingSegmentId && !editing)} />;
        }} /> : undefined}>{null}</ReferencePanel></div>
        {!editingSegmentId && <OperationList operations={sortedOperations} enhance={enhance} colors={operationColors} segmentNumbers={segmentNumbers} hoveredOperationId={hoveredOperationId} onHoverOperation={setHoveredOperationId} onEdit={(operation) => setRequestedEditId(operation.id)} onRemove={(operation) => setOperations((current) => current.filter((item) => item.id !== operation.id))} onRemoveEnhance={() => setEnhance(false)} onClear={() => { setOperations([]); setEnhance(false); }} />}
        <div className="preview-row"><label className={`switch ${editingSegmentId ? "switch--disabled" : ""}`}><input type="checkbox" checked={previewOpen} disabled={Boolean(editingSegmentId)} onChange={(event) => setPreviewOpen(event.target.checked)} /><i /> Preview</label></div>
        {previewOpen && <div className="preview-panel"><label><span>Instruction</span><textarea readOnly value={preview?.instruction || ""} /></label><label><span>Target Transcript</span><textarea readOnly value={preview?.target_text || ""} /></label></div>}
        <footer className="section-footer"><SplitGenerationButton leading={<span>{operations.length + Number(enhance)} operation{operations.length + Number(enhance) === 1 ? "" : "s"}</span>} label="Apply Edit" icon={<WandSparkles size={16} />} disabled={busy || !modelReady || Boolean(editingSegmentId) || !snapshot?.edit_source || snapshot.edit_source.segments.some((segment) => !segment.transcript.trim()) || (!operations.length && !enhance)} open={editConfigOpen} onOpenChange={(open) => { setEditConfigOpen(open); if (open) setTtsConfigOpen(false); }} onClick={() => void generateEdit()}><GeneratorSettingsPanel settings={editSettings} odeMethods={snapshot?.ode_methods || ["euler", "midpoint", "rk4"]} onChange={setEditSettings} speakerGuidanceAvailable={snapshot?.edit_source?.use_xvector ?? false} speakerGuidanceMode="auto" autoSpeakerGuidanceEnabled={editAutoUsesXvector(operations, enhance)} /></SplitGenerationButton></footer>
      </section>

      <section className="workstation-section zero-shot-section"><header className="section-header"><div><Sparkles size={17} /><h2>Zero-shot TTS</h2></div></header>
        <div className="zero-shot-workspace"><ReferencePanel panelRef={promptPair} title="Voice Prompt" destination="prompt" reference={snapshot?.prompt || null} presetValue={promptChoice} presets={promptPresets} optional routed={routedTo === "prompt"} onPresetChange={(value) => void choosePreset("prompt", value)} onRoute={route} onUpload={(file) => requestUpload("prompt", file)} uploadsEnabled={uploadsEnabled}>
          <label className="transcript-field"><span>Prompt Transcript</span><textarea aria-label="Prompt Transcript" value={snapshot?.prompt?.text || ""} disabled={!snapshot?.prompt} placeholder="Enter the transcript paired with Prompt Audio." onChange={(event) => updatePromptTranscript(event.target.value)} /></label>
        </ReferencePanel><label className="text-field text-field--large"><span>Target Transcript</span><textarea aria-label="Target Transcript" value={targetTranscript} onChange={(event) => setTargetTranscript(event.target.value)} placeholder="Enter the transcript to synthesize." /></label></div>
        <footer className="section-footer"><SplitGenerationButton label="Synthesize" icon={<Sparkles size={16} />} disabled={busy || !modelReady || !targetTranscript.trim() || Boolean(snapshot?.prompt && !snapshot.prompt.text.trim())} open={ttsConfigOpen} onOpenChange={(open) => { setTtsConfigOpen(open); if (open) setEditConfigOpen(false); }} onClick={() => void generateTts()}><GeneratorSettingsPanel settings={ttsSettings} odeMethods={snapshot?.ode_methods || ["euler", "midpoint", "rk4"]} onChange={setTtsSettings} speakerGuidanceAvailable={snapshot?.prompt?.use_xvector ?? false} /></SplitGenerationButton></footer>
      </section>
    </main>
    <HistoryDrawer revisions={snapshot?.revisions || []} expanded={historyExpanded} onExpandedChange={setHistoryExpanded} onRoute={route} />
    {libraryIntent && <AudioLibraryDialog sessionId={session} intent={libraryIntent} uploadsEnabled={uploadsEnabled} recognitionAvailable={uploadsEnabled && Boolean(health?.recognition_available)} onCancel={() => setLibraryIntent(null)} onApplied={applyLibrarySelection} />}
    {noiseDialogOpen && snapshot?.edit_source && <NoiseLibraryDialog sessionId={session} reference={snapshot.edit_source} expectedVersion={snapshot.edit_source.composition_version} uploadsEnabled={uploadsEnabled} onCancel={() => setNoiseDialogOpen(false)} onApplied={(next) => { applySegmentSnapshot(next); setNoiseDialogOpen(false); }} />}
    <OnboardingTour uploadsEnabled={uploadsEnabled} />
  </div>;
}

function errorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason);
}
