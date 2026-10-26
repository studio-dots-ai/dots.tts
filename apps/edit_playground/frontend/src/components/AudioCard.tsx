import { useEffect, useRef, useState, type ReactNode } from "react";
import { AudioWaveform, ChartNoAxesColumnIncreasing, Download, GripVertical, MoreHorizontal, Pause, Play, Upload } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import Spectrogram from "wavesurfer.js/plugins/spectrogram";
import type { Destination } from "../types";

export type VisualizationMode = "waveform" | "spectrogram";

const VISUALIZATION_SAMPLE_RATE = 48_000;
const WAVEFORM_HEIGHT = 54;
const SPECTROGRAM_HEIGHT = 168;

export interface AudioCardData {
  sourceId: string;
  audioUrl: string;
  text: string;
}

interface AudioCardProps {
  title: string;
  data: AudioCardData | null;
  compact?: boolean;
  showTranscript?: boolean;
  onUpload?: (file?: File) => void;
  onRoute: (source: string, destination: Destination) => void;
  destination?: Destination;
  draggableAudio?: boolean;
  headerActions?: ReactNode;
  downloadFilename?: string;
}

export const DRAG_TYPE = "application/x-dots-audio";
const PLAY_EVENT = "dots-audio-play";
let playerSequence = 0;

function playerId() {
  playerSequence += 1;
  return `audio-player-${playerSequence}`;
}

export function safeCreateWaveform(create: () => WaveSurfer) {
  try {
    return create();
  } catch {
    return null;
  }
}

export function VisualizationToggle({ mode, disabled = false, onChange }: {
  mode: VisualizationMode;
  disabled?: boolean;
  onChange: (mode: VisualizationMode) => void;
}) {
  const showSpectrogram = mode === "waveform";
  const label = disabled ? "Spectrogram unavailable" : showSpectrogram ? "Show spectrogram" : "Show waveform";
  return <button
    type="button"
    className="icon-button visualization-toggle"
    aria-label={label}
    title={label}
    aria-pressed={mode === "spectrogram"}
    disabled={disabled}
    draggable={false}
    onClick={() => onChange(showSpectrogram ? "spectrogram" : "waveform")}
  >
    {showSpectrogram ? <ChartNoAxesColumnIncreasing size={15} /> : <AudioWaveform size={15} />}
  </button>;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

export function AudioPlayer({ src, compact = false }: { src: string; compact?: boolean }) {
  const audio = useRef<HTMLAudioElement>(null);
  const id = useRef(playerId());
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setCurrent(0);
  }, [src]);

  useEffect(() => {
    const pauseOther = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== id.current) audio.current?.pause();
    };
    window.addEventListener(PLAY_EVENT, pauseOther);
    return () => window.removeEventListener(PLAY_EVENT, pauseOther);
  }, []);

  const toggle = async () => {
    const element = audio.current;
    if (!element) return;
    if (element.paused) {
      try {
        await element.play();
      } catch {
        setPlaying(false);
      }
    } else element.pause();
  };

  return (
    <div className={`audio-player ${compact ? "audio-player--compact" : ""}`}>
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onTimeUpdate={(event) => setCurrent(event.currentTarget.currentTime)}
        onPlay={() => { window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: id.current })); setPlaying(true); }}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <button className="icon-button player-toggle" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
      </button>
      {!compact && (
        <>
          <input
            aria-label="Seek"
            className="audio-seek"
            type="range"
            min={0}
            max={duration || 1}
            step={0.01}
            value={current}
            style={{ "--audio-progress": `${duration ? (current / duration) * 100 : 0}%` } as React.CSSProperties}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (audio.current) audio.current.currentTime = value;
              setCurrent(value);
            }}
          />
          <span className="audio-time">{formatTime(current)} / {formatTime(duration)}</span>
        </>
      )}
    </div>
  );
}

export function AudioVisualizer({ src, mode, onUnavailable }: {
  src: string;
  mode: VisualizationMode;
  onUnavailable?: () => void;
}) {
  const waveformContainer = useRef<HTMLDivElement>(null);
  const spectrogramContainer = useRef<HTMLDivElement>(null);
  const instance = useRef<WaveSurfer | null>(null);
  const spectrogram = useRef<ReturnType<typeof Spectrogram.create> | null>(null);
  const spectrogramCleanups = useRef<Array<() => void>>([]);
  const id = useRef(playerId());
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!waveformContainer.current) return;
    const containerElement = waveformContainer.current;
    setPlaying(false);
    setCurrent(0);
    setDuration(0);
    setFailed(false);
    setReady(false);
    spectrogram.current = null;
    spectrogramCleanups.current = [];
    const wave = safeCreateWaveform(() => WaveSurfer.create({
      container: containerElement,
      url: src,
      sampleRate: VISUALIZATION_SAMPLE_RATE,
      height: WAVEFORM_HEIGHT,
      waveColor: "#c8c8c2",
      progressColor: "#6657c8",
      cursorColor: "transparent",
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
    }));
    if (!wave) {
      setFailed(true);
      onUnavailable?.();
      return;
    }
    instance.current = wave;
    let cleanups: Array<() => void>;
    try {
      cleanups = [
        wave.on("ready", (value) => { setDuration(value); setReady(true); }),
        wave.on("timeupdate", (value) => setCurrent(value)),
        wave.on("play", () => { window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: id.current })); setPlaying(true); }),
        wave.on("pause", () => setPlaying(false)),
        wave.on("finish", () => setPlaying(false)),
        wave.on("error", () => { setFailed(true); onUnavailable?.(); }),
      ];
    } catch {
      wave.destroy();
      instance.current = null;
      setFailed(true);
      onUnavailable?.();
      return;
    }
    return () => {
      spectrogramCleanups.current.forEach((cleanup) => cleanup());
      spectrogramCleanups.current = [];
      spectrogram.current = null;
      cleanups.forEach((cleanup) => cleanup());
      wave.destroy();
      if (instance.current === wave) instance.current = null;
    };
  }, [src]);

  useEffect(() => {
    if (mode !== "spectrogram" || !ready || !instance.current || !spectrogramContainer.current || spectrogram.current) return;
    if (typeof Worker === "undefined") {
      onUnavailable?.();
      return;
    }
    try {
      const plugin = Spectrogram.create({
        container: spectrogramContainer.current,
        sampleRate: VISUALIZATION_SAMPLE_RATE,
        height: SPECTROGRAM_HEIGHT,
        labels: true,
        labelsBackground: "rgba(250, 250, 248, .82)",
        labelsColor: "#686a66",
        labelsHzColor: "#8a8c87",
        fftSamples: 2048,
        frequencyMin: 0,
        frequencyMax: VISUALIZATION_SAMPLE_RATE / 2,
        scale: "linear",
        windowFunc: "hann",
        rangeDB: 80,
        autoGain: true,
        colorMap: "roseus",
        useWebWorker: true,
        fallbackToMainThread: false,
      });
      const fail = () => {
        spectrogramCleanups.current.forEach((cleanup) => cleanup());
        spectrogramCleanups.current = [];
        spectrogram.current = null;
        plugin.destroy();
        onUnavailable?.();
      };
      spectrogramCleanups.current = [
        plugin.on("click", (relativeX) => instance.current?.seekTo(relativeX)),
        plugin.on("error", fail),
      ];
      instance.current.registerPlugin(plugin);
      spectrogram.current = plugin;
    } catch {
      onUnavailable?.();
    }
  }, [mode, onUnavailable, ready]);

  useEffect(() => {
    const pauseOther = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== id.current) instance.current?.pause();
    };
    window.addEventListener(PLAY_EVENT, pauseOther);
    return () => window.removeEventListener(PLAY_EVENT, pauseOther);
  }, []);

  if (failed) return <AudioPlayer src={src} />;
  const progress = duration ? Math.min(100, Math.max(0, current / duration * 100)) : 0;
  return (
    <div className={`waveform-player audio-visualizer audio-visualizer--${mode}`} draggable={false}>
      <button className="icon-button player-toggle" onClick={() => instance.current?.playPause()} aria-label={playing ? "Pause" : "Play"}>
        {playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
      </button>
      <div className="audio-visualizer__viewport">
        <div className="audio-visualizer__layer audio-visualizer__waveform" aria-hidden={mode !== "waveform"}>
          <div className="waveform-canvas" ref={waveformContainer} aria-label="Audio waveform" />
        </div>
        <div className="audio-visualizer__layer audio-visualizer__spectrogram" aria-hidden={mode !== "spectrogram"}>
          <div className="spectrogram-canvas" ref={spectrogramContainer} aria-label="Audio spectrogram from 0 to 24 kHz" />
          <i className="spectrogram-playhead" aria-hidden="true" style={{ left: `${progress}%` }} />
        </div>
      </div>
      <span className="audio-time">{formatTime(current)} / {formatTime(duration)}</span>
    </div>
  );
}

function ServeMenu({ source, destination, onRoute }: { source: string; destination?: Destination; onRoute: AudioCardProps["onRoute"] }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeEscape);
    };
  }, [open]);
  return (
    <div className="serve-menu" ref={root}>
      <button
        className="icon-button"
        aria-label="Serve as"
        title="Serve as"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <MoreHorizontal size={17} />
      </button>
      {open && (
        <div className="menu-panel" role="menu">
          <div className="menu-label">Serve as</div>
          <button disabled={destination === "prompt"} onClick={() => { onRoute(source, "prompt"); setOpen(false); }}>Voice Prompt</button>
          <button disabled={destination === "edit_source"} onClick={() => { onRoute(source, "edit_source"); setOpen(false); }}>Edit Source</button>
        </div>
      )}
    </div>
  );
}

export function AudioCard({
  title,
  data,
  compact = false,
  showTranscript = false,
  onUpload,
  onRoute,
  destination,
  draggableAudio = true,
  headerActions,
  downloadFilename,
}: AudioCardProps) {
  const [mode, setMode] = useState<VisualizationMode>("waveform");
  const [visualizationAvailable, setVisualizationAvailable] = useState(true);
  useEffect(() => setVisualizationAvailable(true), [data?.audioUrl]);
  const unavailable = () => {
    setVisualizationAvailable(false);
    setMode("waveform");
  };
  const startDrag = (event: React.DragEvent) => {
    if (!data) return;
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData(DRAG_TYPE, data.sourceId);
    event.dataTransfer.setData("text/plain", data.sourceId);
  };
  return (
    <article
      className={`audio-card ${compact ? "audio-card--compact" : ""} ${data ? "" : "audio-card--empty"}`}
      draggable={compact && Boolean(data)}
      onDragStart={compact ? startDrag : undefined}
    >
      <header
        className="audio-card__header"
        draggable={draggableAudio && Boolean(data)}
        onDragStart={draggableAudio ? startDrag : undefined}
      >
        {data && draggableAudio && <GripVertical className="drag-grip" size={12} />}
        <span>{title}</span>
        {data && !compact && <VisualizationToggle mode={mode} disabled={!visualizationAvailable} onChange={setMode} />}
        <div className="audio-card__actions" draggable={false}>
          {headerActions}
          {onUpload && !compact && (
            <>
              <button className="icon-button" onClick={() => onUpload()} title="Choose audio" aria-label="Choose audio">
                <Upload size={15} />
              </button>
            </>
          )}
          {data && downloadFilename && <a className="icon-button" href={data.audioUrl} download={downloadFilename} aria-label={`Download ${title}`} title={`Download ${title}`}><Download size={15} /></a>}
          {data && <ServeMenu source={data.sourceId} destination={destination} onRoute={onRoute} />}
        </div>
      </header>
      {data ? (
        <>
          {compact ? <AudioPlayer src={data.audioUrl} compact /> : <AudioVisualizer src={data.audioUrl} mode={mode} onUnavailable={unavailable} />}
          {!compact && showTranscript && <div className="result-transcript"><span>Result Transcript</span><p>{data.text}</p></div>}
        </>
      ) : onUpload ? (
        <button className="audio-dropzone" onClick={() => onUpload()}>
          <Upload size={17} />
          <span>Drop or choose audio</span>
        </button>
      ) : (
        <div className="audio-empty-state">No result yet</div>
      )}
    </article>
  );
}
