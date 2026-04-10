import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.18),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(20,184,166,0.16),_transparent_26%),linear-gradient(135deg,#020617_0%,#0f172a_45%,#022c22_100%)] text-white">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-8">
        <header className="flex items-center justify-between">
          <div className="text-sm font-medium tracking-wide text-white/80">
            NG08 · Emotion Recognition from Speech
          </div>
          <Link
            href="/dashboard"
            className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm text-white/90 backdrop-blur transition hover:bg-white/10"
          >
            Open Dashboard
          </Link>
        </header>

        <section className="flex flex-1 items-center">
          <div className="grid w-full items-center gap-10 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Deep Learning + Real-Time Inference
              </div>

              <h1 className="max-w-3xl text-4xl font-semibold leading-tight sm:text-5xl lg:text-6xl">
                Understand emotion from speech with a polished, real-time web demo.
              </h1>

              <p className="mt-5 max-w-2xl text-base leading-7 text-white/70 sm:text-lg">
                Our capstone project combines pretrained speech representation learning,
                backend inference, and a modern web interface to predict human emotion
                from uploaded or recorded speech audio.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/dashboard"
                  className="rounded-2xl bg-indigo-500 px-6 py-3 text-sm font-medium text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400"
                >
                  Launch Demo
                </Link>
                <a
                  href="#features"
                  className="rounded-2xl border border-white/15 bg-white/5 px-6 py-3 text-sm font-medium text-white/90 backdrop-blur transition hover:bg-white/10"
                >
                  Learn More
                </a>
              </div>

              <div className="mt-10 grid max-w-2xl gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                  <div className="text-2xl font-semibold">8</div>
                  <div className="mt-1 text-sm text-white/60">Emotion classes</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                  <div className="text-2xl font-semibold">WavLM</div>
                  <div className="mt-1 text-sm text-white/60">Transfer learning model</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                  <div className="text-2xl font-semibold">Web App</div>
                  <div className="mt-1 text-sm text-white/60">Mic + upload workflow</div>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute -inset-4 rounded-[2rem] bg-indigo-500/10 blur-3xl" />
              <div className="relative rounded-[2rem] border border-white/10 bg-slate-900/70 p-5 shadow-2xl backdrop-blur">
                <div className="rounded-[1.5rem] border border-white/10 bg-[linear-gradient(180deg,rgba(30,41,59,0.9),rgba(15,23,42,0.95))] p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-white/50">Live demo preview</div>
                      <div className="mt-1 text-lg font-semibold">Emotion Dashboard</div>
                    </div>
                    <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/70">
                      Online
                    </div>
                  </div>

                  <div className="grid gap-4">
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="text-sm text-white/60">Predicted emotion</div>
                      <div className="mt-2 text-3xl font-semibold">Surprised</div>
                      <div className="mt-4 h-2.5 rounded-full bg-white/10">
                        <div className="h-2.5 w-[82%] rounded-full bg-gradient-to-r from-indigo-400 to-emerald-400" />
                      </div>
                      <div className="mt-2 text-right text-xs text-white/60">82% confidence</div>
                    </div>

                    <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                      {[
                        ["Surprised", "82%"],
                        ["Happy", "11%"],
                        ["Fearful", "4%"],
                        ["Neutral", "3%"],
                      ].map(([label, value]) => (
                        <div
                          key={label}
                          className="flex items-center justify-between rounded-xl border border-white/10 bg-black/10 px-4 py-3"
                        >
                          <span className="text-sm font-medium">{label}</span>
                          <span className="text-sm text-white/65">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section
          id="features"
          className="grid gap-4 border-t border-white/10 py-8 text-sm text-white/70 sm:grid-cols-3"
        >
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="mb-2 text-white">Speech-first pipeline</div>
            Upload a file or record directly from the browser microphone for instant inference.
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="mb-2 text-white">Transfer learning backbone</div>
            Pretrained speech representations improve performance over basic CNN baselines.
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="mb-2 text-white">Deployment-ready system</div>
            Frontend UI, backend model API, and real-time prediction output are fully integrated.
          </div>
        </section>
      </div>
    </main>
  );
}