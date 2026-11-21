import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Plot } from "../charts/Plot";
import { Badge, Card, Empty, ErrorState, fmt, KeyValue, Loading, ModalityDot, Notice, PageHeader, pct, titleCase } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api, ApiError } from "../services/api";
import type { Job, Prediction, RecordingDetail, Waveform } from "../types/api";

const HEADLINE = ["speech_rate_wpm", "articulation_rate_syll_per_s", "pause_rate_per_min", "mean_pause_s", "long_pause_rate_per_min",
  "f0_mean_hz", "f0_std_semitones", "f0_range_semitones", "jitter_local", "shimmer_local_db", "hnr_db", "voiced_fraction"];

function UploadPanel() {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function poll(jobId: string, key: string) {
    for (let i = 0; i < 60; i++) {
      const job = await api.get<Job>(`/monitoring/jobs/${jobId}`);
      setStatus(`Processing: ${job.status}`);
      if (job.status === "succeeded") return navigate(`/speech/${key}`);
      if (job.status === "failed") throw new ApiError(500, "job_failed", job.error ?? "Processing failed");
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new ApiError(0, "timeout", "Processing is taking longer than expected; check System Monitoring.");
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.post<{ recording_key: string; job: Job }>("/recordings/upload", new FormData(e.currentTarget));
      setStatus("Uploaded; extracting features…");
      await poll(res.job.id, res.recording_key);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Upload a recording" subtitle="WAV / FLAC / MP3 / OGG. Raw audio is deleted after feature extraction unless retention is enabled.">
      <form onSubmit={submit} className="grid gap-3 md:grid-cols-2">
        <label className="text-sm"><span className="eyebrow">Audio file</span><input name="file" type="file" accept=".wav,.flac,.mp3,.ogg,audio/*" required className="input mt-1" /></label>
        <label className="text-sm"><span className="eyebrow">Participant pseudonym</span><input name="participant" required pattern="[A-Za-z0-9_\-]{3,64}" placeholder="e.g. STUDY-0042" className="input mt-1" /></label>
        <label className="text-sm"><span className="eyebrow">Session date</span><input name="session_date" type="date" className="input mt-1" /></label>
        <label className="text-sm"><span className="eyebrow">Task</span><input name="task" placeholder="picture_description" className="input mt-1" /></label>
        <label className="text-sm md:col-span-2"><span className="eyebrow">Transcript (optional, participant speech only)</span><textarea name="transcript" rows={2} className="input mt-1" /></label>
        <label className="flex items-start gap-2 text-sm md:col-span-2">
          <input type="checkbox" name="consent_confirmed" value="true" required className="mt-1" />
          <span>I confirm documented research consent exists for this recording and that the pseudonym contains no identifying information.</span>
        </label>
        <div className="flex items-center gap-3 md:col-span-2">
          <button className="btn-primary" disabled={busy} type="submit">{busy ? "Working…" : "Upload & analyse"}</button>
          {status && <span className="text-sm text-ink-muted">{status}</span>}
        </div>
        {error && <div className="md:col-span-2"><Notice tone="alert">{error}</Notice></div>}
      </form>
    </Card>
  );
}

function RecordingPicker() {
  const navigate = useNavigate();
  const [key, setKey] = useState("");
  const recent = useApi<Prediction[]>("/predictions?limit=12");
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <UploadPanel />
      <Card title="Open a recording">
        <form className="mb-4 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (key.trim()) navigate(`/speech/${key.trim()}`); }}>
          <input className="input" placeholder="Recording key, e.g. SYN-P001-S01-R01" value={key} onChange={(e) => setKey(e.target.value)} />
          <button className="btn-ghost" type="submit">Open</button>
        </form>
        <div className="eyebrow mb-2">Recently scored</div>
        {recent.loading ? <Loading /> : recent.error ? <ErrorState error={recent.error} /> : !recent.data?.length ? <Empty>No predictions yet.</Empty> : (
          <ul className="divide-y divide-line text-sm">
            {recent.data.map((p) => (
              <li key={p.id} className="flex items-center justify-between py-1.5">
                <Link to={`/speech/${p.recording_key}`} className="font-mono text-brand-700 hover:underline">{p.recording_key}</Link>
                <span className="flex items-center gap-2"><Badge>{p.split ?? p.origin}</Badge>{p.abstain && <Badge tone="warn">review</Badge>}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function WaveformPanel({ recordingKey }: { recordingKey: string }) {
  const { data, error, loading } = useApi<Waveform>(`/recordings/${recordingKey}/waveform`);
  if (loading) return <Loading label="Loading waveform" />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;
  const t = data.max.map((_, i) => i * data.time_step_s);
  const shapes = [
    ...data.speech_segments.map(([s, e]) => ({ type: "rect" as const, xref: "x" as const, yref: "paper" as const, x0: s, x1: e, y0: 0, y1: 1, fillcolor: "rgba(15,118,110,0.07)", line: { width: 0 } })),
    ...data.pauses.filter((p) => p.duration >= data.long_pause_threshold_s).map((p) => ({ type: "rect" as const, xref: "x" as const, yref: "paper" as const, x0: p.start, x1: p.end, y0: 0, y1: 1, fillcolor: "rgba(217,119,6,0.16)", line: { width: 0 } })),
  ];
  const pauseDur = data.pauses.map((p) => p.duration);
  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <div className="xl:col-span-2">
        <Plot height={230} data={[
          { x: t, y: data.max, type: "scatter", mode: "lines", line: { width: 0.6, color: "#0f766e" }, hoverinfo: "skip", showlegend: false },
          { x: t, y: data.min, type: "scatter", mode: "lines", fill: "tonexty", fillcolor: "rgba(15,118,110,0.55)", line: { width: 0.6, color: "#0f766e" }, hoverinfo: "skip", showlegend: false },
        ]} layout={{ shapes, xaxis: { title: { text: "Time (s)" } }, yaxis: { range: [-1, 1], showticklabels: false }, margin: { l: 16, r: 8, t: 4, b: 40 } }} />
        <p className="mt-1 flex flex-wrap gap-4 text-xs text-ink-muted">
          <span><span className="mr-1 inline-block h-2.5 w-3 bg-brand-600/20 align-middle" />Detected speech</span>
          <span><span className="mr-1 inline-block h-2.5 w-3 bg-amber-500/30 align-middle" />Long pause (≥ {data.long_pause_threshold_s}s)</span>
        </p>
      </div>
      <div>
        <div className="eyebrow mb-1">Pause durations ({data.pauses.length})</div>
        {pauseDur.length ? <Plot height={200} data={[{ type: "histogram", x: pauseDur, nbinsx: 15, marker: { color: "#d97706" } }]}
          layout={{ xaxis: { title: { text: "Seconds" } }, yaxis: { title: { text: "Count" } }, margin: { l: 40, r: 8, t: 4, b: 40 }, bargap: 0.05 }} />
          : <Empty>No pauses detected.</Empty>}
      </div>
    </div>
  );
}

export function SpeechAnalysis() {
  const { key } = useParams();
  const rec = useApi<RecordingDetail>(key ? `/recordings/${key}` : null);
  if (!key) return (<><PageHeader title="Speech analysis" description="Upload a permitted recording or open an existing one to inspect waveform, pauses, transcript and features." /><RecordingPicker /></>);
  if (rec.loading) return <Loading label="Loading recording" />;
  if (rec.error) return <ErrorState error={rec.error} onRetry={rec.reload} />;
  const r = rec.data!;
  const f = r.features;
  const flags = r.quality?.flags ?? [];
  const p = r.latest_prediction;
  const featureRows = Object.entries(f).filter(([k]) => r.feature_modalities[k] !== "descriptive" && !k.startsWith("mfcc"));

  return (
    <>
      <PageHeader title={r.recording_key} description={<>Participant <Link className="font-mono text-brand-700 hover:underline" to={`/participants/${r.participant}`}>{r.participant}</Link> · session S{r.session_index + 1}{r.session_date ? ` · ${r.session_date}` : ""}</>}>
        <Link className="btn-ghost" to="/speech">Other recordings</Link>
        <Link className="btn-primary" to={`/explainability/${r.recording_key}`}>Explain prediction</Link>
      </PageHeader>

      {r.quality_usable === false && <div className="mb-4"><Notice tone="alert" title="Recording failed quality checks">Flags: {flags.join(", ") || "unknown"}. Features and predictions are withheld for unusable audio. {r.quality?.extraction_error}</Notice></div>}
      {r.quality_usable !== false && flags.length > 0 && <div className="mb-4"><Notice title="Data quality warnings">{flags.map(titleCase).join(", ")} — interpret features with caution.</Notice></div>}

      <div className="grid gap-4 md:grid-cols-4">
        <div className="card card-pad"><div className="eyebrow">Duration</div><div className="num mt-2 text-2xl font-semibold">{r.duration_s ? `${r.duration_s.toFixed(1)} s` : "—"}</div><div className="mt-1 text-xs text-ink-muted">analysed {fmt(f.analysed_duration_s, 1)} s of participant speech</div></div>
        <div className="card card-pad"><div className="eyebrow">Speech rate</div><div className="num mt-2 text-2xl font-semibold">{fmt(f.speech_rate_wpm, 0)} <span className="text-sm font-normal text-ink-muted">wpm</span></div><div className="mt-1 text-xs text-ink-muted">{fmt(f.speech_rate_syll_per_s, 2)} syllables/s (acoustic)</div></div>
        <div className="card card-pad"><div className="eyebrow">Pauses</div><div className="num mt-2 text-2xl font-semibold">{fmt(f.pause_rate_per_min, 1)} <span className="text-sm font-normal text-ink-muted">/min</span></div><div className="mt-1 text-xs text-ink-muted">mean {fmt(f.mean_pause_s, 2)} s · long {fmt(f.long_pause_rate_per_min, 1)}/min</div></div>
        <div className="card card-pad"><div className="eyebrow">Audio quality</div><div className="num mt-2 text-2xl font-semibold">{fmt(r.quality?.snr_db, 0)} <span className="text-sm font-normal text-ink-muted">dB SNR est.</span></div><div className="mt-1 text-xs text-ink-muted">clipping {pct(r.quality?.clipping_fraction, 2)} · silence {pct(r.quality?.silence_fraction)}</div></div>
      </div>

      <Card title="Waveform and pause analysis" className="mt-6"><WaveformPanel recordingKey={r.recording_key} /></Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card title="Transcript" subtitle={`Source: ${r.transcript_source}${r.has_speaker_turns ? " · speaker turns available" : ""}`} className="xl:col-span-2">
          {r.transcript ? <p className="max-h-64 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-ink-soft">{r.transcript}</p>
            : <Empty>No transcript. Linguistic features are missing for this recording (enable Whisper transcription to compute them).</Empty>}
          {r.transcript_source === "whisper" && <p className="mt-2 text-xs text-ink-muted">ASR transcripts may omit disfluencies; filler and repetition rates are not comparable with manual transcripts.</p>}
        </Card>
        <Card title="Model output" subtitle="Decision support only">
          {p ? (
            <div className="space-y-3 text-sm">
              <div className="num text-3xl font-semibold">{fmt(p.probability, 2)}</div>
              <div className="text-ink-muted">Calibrated probability of the dataset's “cognitive impairment” label pattern</div>
              {p.abstain ? <Notice title="Refer for clinician review">{((p.uncertainty?.reasons as string[]) ?? []).join("; ") || "Uncertain prediction"}</Notice>
                : <Badge tone="neutral">{titleCase(p.model_label)}</Badge>}
              <p className="text-xs text-ink-muted">{p.disclaimer}</p>
            </div>
          ) : <Empty>No prediction for this recording.</Empty>}
        </Card>
      </div>

      <Card title="Extracted features" subtitle="Measurement outputs, not validated clinical indicators" className="mt-6">
        <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
          {HEADLINE.filter((k) => k in f).map((k) => (
            <div key={k} className="rounded-lg bg-canvas px-3 py-2"><div className="truncate text-[11px] text-ink-muted" title={k}>{titleCase(k)}</div><div className="num font-semibold">{fmt(f[k], 3)}</div></div>
          ))}
        </div>
        <details>
          <summary className="cursor-pointer text-sm font-medium text-brand-700">All {featureRows.length} features</summary>
          <div className="mt-3 grid gap-x-8 md:grid-cols-2 xl:grid-cols-3">
            {featureRows.map(([k, v]) => (
              <div key={k} className="flex items-center justify-between border-b border-line/60 py-1 text-sm">
                <span className="flex items-center gap-1.5 text-ink-soft"><ModalityDot modality={r.feature_modalities[k]} />{k}</span>
                <span className="num">{fmt(v, 3)}</span>
              </div>
            ))}
          </div>
        </details>
        <div className="mt-4"><KeyValue items={[["Raw audio retained", r.raw_audio_retained ? "yes" : "no (deleted after processing)"], ["Split", r.split ?? "—"], ["Native sample rate", r.sample_rate ? `${r.sample_rate} Hz` : "—"]]} /></div>
      </Card>
    </>
  );
}
