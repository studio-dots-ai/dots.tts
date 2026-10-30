import type { ReactNode, Ref } from "react";
import type { AudioReference, Destination } from "../types";
import { AudioCard, type AudioCardData } from "./AudioCard";
import { ReferenceDropZone } from "./ReferenceDropZone";

interface Preset {
  name: string;
}

const PRESET_LABELS: Record<string, string> = {
  text_en: "Text · English",
  text_zh: "Text · Mandarin",
  emotion_en: "Emotion · English",
  pitch_zh: "Pitch · Mandarin",
  rate_zh: "Rate · Mandarin",
  pause_en: "Pause · English",
  compositional_zh: "Compositional · Mandarin",
  noise_preservation_en: "Noise preservation · English",
  denoising_zh: "Denoising · Mandarin",
  multi_speaker_zh: "Multi-speaker · Mandarin",
  female_en: "Female · English",
  female_zh: "Female · Mandarin",
  male_en: "Male · English",
  male_zh: "Male · Mandarin",
  child: "Child · Mandarin",
  genshin_female: "Character · Female",
  genshin_male: "Character · Male",
  taozi: "Long-form · Mandarin",
};

interface Props {
  panelRef?: Ref<HTMLDivElement>;
  title: "Edit Source" | "Voice Prompt";
  destination: Destination;
  reference: AudioReference | null;
  presetValue: string;
  presets: Preset[];
  optional?: boolean;
  routed?: boolean;
  children: ReactNode;
  onPresetChange: (value: string) => void;
  onRoute: (source: string, destination: Destination) => void;
  onUpload: (file?: File) => void;
  uploadsEnabled?: boolean;
  audioContent?: ReactNode;
}

function audioData(reference: AudioReference | null): AudioCardData | null {
  return reference ? { sourceId: reference.source_id, audioUrl: reference.audio_url, text: reference.text } : null;
}

export function PresetSelect({ value, presets, includeRandom = false, onChange }: {
  value: string;
  presets: Preset[];
  includeRandom?: boolean;
  onChange: (value: string) => void;
}) {
  return <label className="preset-select"><span>Presets</span><select value={value} onChange={(event) => onChange(event.target.value)}>{includeRandom && <option value="__random__">Random</option>}<option value="__custom__" disabled>Custom / Revision</option>{presets.map((preset) => <option key={preset.name} value={preset.name}>{PRESET_LABELS[preset.name] || preset.name}</option>)}</select></label>;
}

export function ReferencePanel({
  panelRef,
  title,
  destination,
  reference,
  presetValue,
  presets,
  optional = false,
  routed = false,
  children,
  onPresetChange,
  onRoute,
  onUpload,
  uploadsEnabled = true,
  audioContent,
}: Props) {
  return (
    <ReferenceDropZone
      ref={panelRef}
      destination={destination}
      className={`paired-block reference-panel ${routed ? "paired-block--routed" : ""}`}
      onRoute={onRoute}
      onUpload={onUpload}
      uploadsEnabled={uploadsEnabled}
    >
      <div className="pair-heading">
        <span>{title}{optional && <small>Optional</small>}</span>
        <PresetSelect includeRandom={optional} value={presetValue} presets={presets} onChange={onPresetChange} />
      </div>
      {audioContent || <AudioCard
        title={destination === "prompt" ? "Prompt Audio" : "Source Audio"}
        data={audioData(reference)}
        destination={destination}
        onUpload={uploadsEnabled ? onUpload : undefined}
        onRoute={onRoute}
        downloadFilename={destination === "prompt" ? "dots-tts-prompt-audio.wav" : "dots-tts-source-audio.wav"}
      />}
      {children}
    </ReferenceDropZone>
  );
}
