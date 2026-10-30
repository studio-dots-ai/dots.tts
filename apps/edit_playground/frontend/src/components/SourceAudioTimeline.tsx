import { Fragment, type ReactNode, useState } from "react";
import { GripVertical, Plus, Wind, X } from "lucide-react";
import type { AudioReference, Destination } from "../types";
import { AudioCard, DRAG_TYPE } from "./AudioCard";

export const SEGMENT_DRAG_TYPE = "application/x-dots-audio-segment";
export type SegmentDropZone = "before" | "replace" | "after";

export function classifySegmentDrop(clientX: number, left: number, width: number): SegmentDropZone {
  const fraction = width > 0 ? (clientX - left) / width : 0.5;
  return fraction < 1 / 3 ? "before" : fraction > 2 / 3 ? "after" : "replace";
}

export function segmentDropIntent(zone: SegmentDropZone, internal: boolean, canInsert = true) {
  if (internal) return zone === "replace" ? "none" : "move";
  if (!canInsert && zone !== "replace") return "none";
  return zone === "replace" ? "replace" : "insert";
}

function payload(event: React.DragEvent) {
  const segmentId = event.dataTransfer.getData(SEGMENT_DRAG_TYPE);
  const sourceId = event.dataTransfer.getData(DRAG_TYPE);
  return { segmentId, sourceId, file: event.dataTransfer.files[0] };
}

export function SourceAudioTimeline({ reference, onRoute, onInsert, onMove, onReplace, onRemove, onUploadWhole, onUploadAt, onUploadReplace, uploadsEnabled = true, onNoise, enhance, onEnhanceToggle, globalHint, renderTranscript }: {
  reference: AudioReference;
  onRoute: (source: string, destination: Destination) => void;
  onInsert: (sourceId: string, index: number) => void;
  onMove: (segmentId: string, index: number) => void;
  onReplace: (segmentId: string, sourceId: string) => void;
  onRemove: (segmentId: string) => void;
  onUploadWhole: (file?: File) => void;
  onUploadAt: (index: number, file?: File) => void;
  onUploadReplace: (segmentId: string, file?: File) => void;
  uploadsEnabled?: boolean;
  onNoise?: () => void;
  enhance: boolean;
  onEnhanceToggle: () => void;
  globalHint?: ReactNode;
  renderTranscript: (segmentId: string, index: number) => ReactNode;
}) {
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const canInsert = reference.segments.length < 3;
  const multipleSegments = reference.segments.length > 1;
  const showFullSource = multipleSegments || Boolean(reference.noise);
  const noiseButton = () => onNoise && <button className={`icon-button noise-button noise-button--global ${reference.noise ? "noise-button--active" : ""}`} aria-label={reference.noise ? `Edit background noise, ${reference.noise.snr_db} dB` : "Add background noise to Full Source Audio"} title={reference.noise ? `Noise · ${reference.noise.snr_db} dB` : "Add background noise"} onClick={() => onNoise()}>{!reference.noise && <span className="noise-button__plus">+</span>}<Wind size={15} />{reference.noise && <span>{reference.noise.snr_db}</span>}</button>;
  const sourceControls = <div className="source-global-controls"><button className={`enhance-operation ${enhance ? "enhance-operation--active" : ""}`} aria-pressed={enhance} onClick={onEnhanceToggle}><Wind size={13} /> Enhance</button><div className="segment-summary">{reference.segments.length} segment{reference.segments.length === 1 ? "" : "s"} · {reference.duration_seconds.toFixed(1)}s / 40s</div></div>;
  const dropAt = (event: React.DragEvent, index: number) => {
    event.preventDefault(); event.stopPropagation(); setDropTarget(null);
    const value = payload(event);
    if (value.file && uploadsEnabled) onUploadAt(index, value.file);
    else if (value.segmentId) onMove(value.segmentId, index);
    else if (value.sourceId) onInsert(value.sourceId, index);
  };
  const gap = (index: number) => canInsert ? (
    <button className={`segment-insert ${dropTarget === `gap-${index}` ? "segment-insert--active" : ""}`} aria-label={`Insert audio at position ${index + 1}`} onClick={() => onUploadAt(index)} onDragEnter={(event) => { event.preventDefault(); event.stopPropagation(); setDropTarget(`gap-${index}`); }} onDragOver={(event) => { event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = "copy"; setDropTarget(`gap-${index}`); }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropTarget(null); }} onDrop={(event) => dropAt(event, index)}><Plus size={14} /></button>
  ) : null;
  return (
    <div className="source-audio-timeline">
      {showFullSource && <AudioCard title="Full Source Audio" data={{ sourceId: "edit_source", audioUrl: reference.audio_url, text: reference.text }} destination="edit_source" onUpload={uploadsEnabled ? onUploadWhole : undefined} onRoute={onRoute} headerActions={noiseButton()} downloadFilename="dots-tts-source-audio.wav" />}
      {multipleSegments && globalHint}
      <div className="segments-strip">
        {gap(0)}
        {reference.segments.map((segment, index) => <Fragment key={segment.id}><div className="segment-shell" key={`${segment.id}:${segment.audio_url}`}>
          <div
          className={`segment-audio-drop-target ${dropTarget?.startsWith(`${segment.id}:`) ? `segment-shell--drop-${dropTarget.split(":")[1]}` : ""}`}
          onDragEnter={(event) => {
            if (!uploadsEnabled && event.dataTransfer.types.includes("Files") && !event.dataTransfer.types.includes(DRAG_TYPE) && !event.dataTransfer.types.includes(SEGMENT_DRAG_TYPE)) return;
            event.preventDefault(); event.stopPropagation();
          }}
          onDragOver={(event) => {
            if (!uploadsEnabled && event.dataTransfer.types.includes("Files") && !event.dataTransfer.types.includes(DRAG_TYPE) && !event.dataTransfer.types.includes(SEGMENT_DRAG_TYPE)) {
              event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = "none"; setDropTarget(null); return;
            }
            event.preventDefault(); event.stopPropagation();
            const rect = event.currentTarget.getBoundingClientRect();
            const zone = classifySegmentDrop(event.clientX, rect.left, rect.width);
            const intent = segmentDropIntent(zone, Boolean(event.dataTransfer.getData(SEGMENT_DRAG_TYPE)), canInsert);
            event.dataTransfer.dropEffect = intent === "none" ? "none" : intent === "move" ? "move" : "copy";
            setDropTarget(intent === "none" ? null : `${segment.id}:${zone}`);
          }}
          onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDropTarget(null); }}
          onDrop={(event) => {
            event.preventDefault(); event.stopPropagation(); setDropTarget(null);
            const rect = event.currentTarget.getBoundingClientRect();
            const zone = classifySegmentDrop(event.clientX, rect.left, rect.width);
            const value = payload(event);
            if (value.file && !uploadsEnabled) return;
            const intent = segmentDropIntent(zone, Boolean(value.segmentId), canInsert);
            if (intent === "move" || intent === "insert") return dropAt(event, zone === "before" ? index : index + 1);
            if (intent === "none") return;
            if (value.file) onUploadReplace(segment.id, value.file);
            else if (value.sourceId) onReplace(segment.id, value.sourceId);
          }}
        >
          {reference.segments.length > 1 && <div className="segment-left-actions"><span className="segment-grip" draggable onDragStart={(event) => { event.dataTransfer.effectAllowed = "copyMove"; event.dataTransfer.setData(SEGMENT_DRAG_TYPE, segment.id); event.dataTransfer.setData(DRAG_TYPE, segment.source_id); }} aria-label={`Move Segment ${index + 1}`}><GripVertical size={14} /></span><button className="icon-button" aria-label={`Remove Segment ${index + 1}`} onClick={() => onRemove(segment.id)}><X size={14} /></button></div>}
          <AudioCard title={reference.segments.length === 1 ? "Source Segment" : `Segment ${index + 1}`} data={{ sourceId: segment.source_id, audioUrl: segment.audio_url, text: segment.transcript }} destination="edit_source" draggableAudio={reference.segments.length === 1} onUpload={uploadsEnabled ? (reference.segments.length === 1 ? onUploadWhole : (file) => onUploadReplace(segment.id, file)) : undefined} onRoute={onRoute} headerActions={!showFullSource ? noiseButton() : undefined} downloadFilename={`dots-tts-source-segment-${index + 1}.wav`} />
          </div>
          {!multipleSegments && globalHint}
          <div className="segment-transcript">{renderTranscript(segment.id, index)}</div>
          {!multipleSegments && sourceControls}
        </div>{gap(index + 1)}</Fragment>)}
      </div>
      {multipleSegments && sourceControls}
    </div>
  );
}
