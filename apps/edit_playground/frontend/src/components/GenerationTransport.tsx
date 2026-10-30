import type { AudioCardData } from "./AudioCard";
import { AudioCard } from "./AudioCard";
import type { Destination, GenerationEvent } from "../types";

interface Props {
  busy: boolean;
  status: GenerationEvent;
  title: string;
  data: AudioCardData | null;
  onRoute: (source: string, destination: Destination) => void;
}

export function GenerationTransport({ busy, status, title, data, onRoute }: Props) {
  const progress = status.progress == null ? undefined : `${Math.round(status.progress * 100)}%`;
  const downloadFilename = title === "Latest Edit" ? "dots-tts-latest-edit.wav" : "dots-tts-latest-result.wav";
  return <section className={`transport ${busy ? "transport--busy" : ""}`}>{busy ? <div className="generation-progress"><div className="progress-row"><span>{status.phase === "queued" && status.message ? status.message : status.phase}</span><div className={`progress-track ${status.progress == null ? "progress-track--indeterminate" : ""}`}><i style={progress ? { width: progress } : undefined} /></div></div></div> : <AudioCard title={title} data={data} showTranscript onRoute={onRoute} downloadFilename={downloadFilename} />}</section>;
}
