"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

type ProbItem = { label: string; prob: number };
type PredictResponse = {
  topEmotion: string;
  confidence: number;
  probabilities: ProbItem[];
};

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:5000";

function pct(x: number) {
  return Math.round(x * 100);
}

function pretty(label: string) {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export default function DashboardPage() {
  const [mode, setMode] = useState<"upload" | "mic">("mic");
  const [backendOnline, setBackendOnline] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [recording, setRecording] = useState(false);
  const [predicting, setPredicting] = useState(false);

  const [result, setResult] = useState<PredictResponse | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const ping = async () => {
      try {
        const r = await fetch(`${BACKEND}/health`);
        setBackendOnline(r.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    ping();
    const t = setInterval(ping, 2500);
    return () => clearInterval(t);
  }, []);

  const topList = useMemo(() => {
    if (!result) return [];
    return result.probabilities.slice(0, 4);
  }, [result]);

  const clearAudio = () => {
    setFile(null);
    setResult(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl("");
    recordedChunksRef.current = [];
  };

  const onUpload = (f: File) => {
    clearAudio();
    setFile(f);
    setAudioUrl(URL.createObjectURL(f));
  };

  const startRecording = async () => {
    clearAudio();
    setResult(null);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    mediaRecorderRef.current = mr;
    recordedChunksRef.current = [];

    mr.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data);
    };

    mr.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: "audio/webm" });
      const f = new File([blob], "recording.webm", { type: "audio/webm" });
      setFile(f);
      setAudioUrl(URL.createObjectURL(blob));
      stream.getTracks().forEach((t) => t.stop());
    };

    mr.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const predict = async () => {
    if (!file) return;
    setPredicting(true);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const r = await fetch(`${BACKEND}/predict`, {
        method: "POST",
        body: form,
      });

      if (!r.ok) throw new Error("Predict failed");
      const data = (await r.json()) as PredictResponse;
      setResult(data);
    } catch {
      alert("Prediction failed. Is the backend running on port 5000?");
    } finally {
      setPredicting(false);
    }
  };

  const topEmotion = result ? pretty(result.topEmotion) : "—";
  const confidence = result ? pct(result.confidence) : 0;
  const isUncertain = result ? result.confidence < 0.4 : false;

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.18),_transparent_25%),radial-gradient(circle_at_bottom_right,_rgba(20,184,166,0.16),_transparent_25%),linear-gradient(135deg,#020617_0%,#0f172a_45%,#022c22_100%)] text-white">
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-white/45">
              NG08 Capstone Demo
            </div>
            <h1 className="mt-1 text-2xl font-semibold">Emotion Recognition Dashboard</h1>
          </div>

          <div className="flex items-center gap-3">
            <div className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/75">
              {backendOnline ? "Backend online" : "Backend offline"}
            </div>
            <Link
              href="/"
              className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/85 transition hover:bg-white/10"
            >
              Back
            </Link>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
          {/* Left panel */}
          <section className="rounded-3xl border border-white/10 bg-slate-900/60 p-5 shadow-2xl backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-3xl font-semibold leading-tight">
                  Emotion Recognition from <br /> Speech
                </h2>
                <p className="mt-3 max-w-xl text-sm leading-6 text-white/65">
                  Upload an audio file or record from your microphone, then get a predicted
                  emotion and confidence score in real time.
                </p>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/70">
                <div className="flex items-center gap-2">
                  <span className={`h-2.5 w-2.5 rounded-full ${backendOnline ? "bg-emerald-400" : "bg-rose-400"}`} />
                  {backendOnline ? "Backend online" : "Backend offline"}
                </div>
              </div>
            </div>

            <div className="mt-6 inline-flex rounded-2xl bg-black/20 p-1 ring-1 ring-white/10">
              <button
                onClick={() => setMode("upload")}
                className={`rounded-xl px-5 py-2.5 text-sm transition ${
                  mode === "upload" ? "bg-white/10 text-white" : "text-white/60 hover:text-white"
                }`}
              >
                Upload File
              </button>
              <button
                onClick={() => setMode("mic")}
                className={`rounded-xl px-5 py-2.5 text-sm transition ${
                  mode === "mic" ? "bg-indigo-500/40 text-white" : "text-white/60 hover:text-white"
                }`}
              >
                Microphone
              </button>
            </div>

            <div className="mt-5 rounded-3xl border border-white/10 bg-black/15 p-4">
              <div className="flex flex-wrap gap-3">
                {mode === "upload" ? (
                  <label className="cursor-pointer rounded-2xl bg-white/10 px-5 py-3 text-sm ring-1 ring-white/10 transition hover:bg-white/15">
                    <input
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) onUpload(f);
                      }}
                    />
                    Choose File
                  </label>
                ) : (
                  <>
                    <button
                      onClick={startRecording}
                      disabled={recording}
                      className="rounded-2xl bg-white/10 px-5 py-3 text-sm ring-1 ring-white/10 transition hover:bg-white/15 disabled:opacity-50"
                    >
                      Start Recording
                    </button>
                    <button
                      onClick={stopRecording}
                      disabled={!recording}
                      className="rounded-2xl bg-rose-500/50 px-5 py-3 text-sm ring-1 ring-white/10 transition hover:bg-rose-500/60 disabled:opacity-50"
                    >
                      Stop
                    </button>
                  </>
                )}

                <button
                  onClick={predict}
                  disabled={!file || predicting || !backendOnline}
                  className="ml-auto rounded-2xl bg-indigo-500 px-8 py-3 text-sm font-medium ring-1 ring-white/10 transition hover:bg-indigo-400 disabled:opacity-50"
                >
                  {predicting ? "Predicting..." : "Predict"}
                </button>
              </div>

              <div className="mt-4">
                <audio controls className="w-full" src={audioUrl || undefined} />
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <button
                  onClick={clearAudio}
                  className="rounded-2xl bg-white/10 px-5 py-3 text-sm ring-1 ring-white/10 transition hover:bg-white/15"
                >
                  Discard
                </button>

                <div className="text-xs text-white/55">
                  Backend endpoint:{" "}
                  <span className="font-medium text-white/75">{BACKEND}/predict</span>
                </div>
              </div>

              <p className="mt-4 text-sm text-white/50">
                For best results, record 2–5 seconds of clear speech with minimal background noise.
              </p>
            </div>
          </section>

          {/* Right panel */}
          <aside className="rounded-3xl border border-white/10 bg-slate-900/60 p-5 shadow-2xl backdrop-blur">
            <div className="text-sm text-white/55">Prediction</div>

            <div className="mt-3 text-5xl font-semibold tracking-tight">
              {isUncertain ? "Uncertain" : topEmotion}
            </div>

            <div className="mt-5 flex items-center justify-between text-sm text-white/65">
              <span>Confidence</span>
              <span>{result ? `${confidence}%` : "—"}</span>
            </div>

            <div className="mt-2 h-3 w-full rounded-full bg-white/10 ring-1 ring-white/10">
              <div
                className="h-3 rounded-full bg-gradient-to-r from-indigo-400 via-sky-400 to-emerald-400 transition-all"
                style={{ width: `${confidence}%` }}
              />
            </div>

            <div className="mt-5 space-y-3">
              {topList.length === 0 ? (
                <div className="rounded-2xl border border-white/10 bg-black/10 p-4 text-sm text-white/55">
                  Model probabilities will show here after you predict.
                </div>
              ) : (
                topList.map((p) => (
                  <div
                    key={p.label}
                    className="flex items-center justify-between rounded-2xl border border-white/10 bg-black/10 px-4 py-4"
                  >
                    <span className="text-lg font-medium">{pretty(p.label)}</span>
                    <span className="text-lg text-white/65">{pct(p.prob)}%</span>
                  </div>
                ))
              )}
            </div>

            <div className="mt-5 text-sm text-white/50">
              Model probabilities shown below (highest first).
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}