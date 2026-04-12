import { writeFile, unlink } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type PythonResult = {
  id?: number;
  type?: string;
  ok: boolean;
  data?: unknown;
  error?: string;
  traceback?: string;
};

const workspaceRoot = path.resolve(process.cwd(), "..");
const workerScript = path.join(process.cwd(), "server", "predict_worker.py");

let worker: ChildProcessWithoutNullStreams | null = null;
let nextId = 1;
let lineBuffer = "";

function normalizeApiKey(raw: string): string {
  const trimmed = raw.trim().replace(/^['\"]|['\"]$/g, "");
  return trimmed.replace(/^Bearer\s+/i, "").replace(/\s+/g, "");
}

const pending = new Map<
  number,
  {
    resolve: (value: PythonResult) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }
>();

function rejectAllPending(message: string) {
  for (const [id, entry] of pending.entries()) {
    clearTimeout(entry.timeout);
    entry.reject(new Error(`${message} (request ${id})`));
    pending.delete(id);
  }
}

function shutdownWorker() {
  if (worker && !worker.killed) {
    worker.kill();
  }
  worker = null;
  lineBuffer = "";
}

function ensureWorker() {
  if (worker && !worker.killed) {
    return worker;
  }

  const pythonExe = process.env.PYTHON_EXECUTABLE || "python";
  worker = spawn(pythonExe, [workerScript], { cwd: workspaceRoot });
  lineBuffer = "";

  worker.stdout.on("data", (chunk) => {
    lineBuffer += chunk.toString();
    const lines = lineBuffer.split(/\r?\n/);
    lineBuffer = lines.pop() || "";

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || !line.startsWith("{")) {
        continue;
      }

      try {
        const parsed = JSON.parse(line) as PythonResult;
        if (parsed.type === "ready") {
          continue;
        }

        if (typeof parsed.id !== "number") {
          continue;
        }

        const job = pending.get(parsed.id);
        if (!job) {
          continue;
        }

        clearTimeout(job.timeout);
        pending.delete(parsed.id);
        job.resolve(parsed);
      } catch {
        // Ignore malformed worker lines.
      }
    }
  });

  worker.stderr.on("data", () => {
    // stderr is intentionally ignored here; failures are returned in worker JSON responses.
  });

  worker.on("exit", () => {
    worker = null;
    rejectAllPending("Python worker exited");
  });

  worker.on("error", (err) => {
    rejectAllPending(`Python worker error: ${err.message}`);
  });

  return worker;
}

function runWorkerJob(payload: Omit<Record<string, unknown>, "id">): Promise<PythonResult> {
  const proc = ensureWorker();
  const id = nextId++;

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Prediction timed out"));
    }, 240000);

    pending.set(id, { resolve, reject, timeout });
    proc.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
  });
}

export async function POST(request: NextRequest) {
  const form = await request.formData();
  const image = form.get("image");
  const latRaw = form.get("lat");
  const lonRaw = form.get("lon");
  const apiKey = normalizeApiKey(String(form.get("apiKey") || ""));
  const confidence = Number(form.get("confidence"));
  const showVegetation = String(form.get("showVegetation") || "true");
  const showRuins = String(form.get("showRuins") || "true");
  const showStructures = String(form.get("showStructures") || "true");
  const showBoulders = String(form.get("showBoulders") || "true");
  const showOthers = String(form.get("showOthers") || "true");
  const useAiInsight = String(form.get("useAiInsight") || "false");

  if (!(image instanceof File)) {
    return NextResponse.json({ error: "Image file is required" }, { status: 400 });
  }

  const lat = Number(latRaw);
  const lon = Number(lonRaw);

  if (Number.isNaN(lat) || Number.isNaN(lon)) {
    return NextResponse.json({ error: "Latitude and longitude must be valid numbers" }, { status: 400 });
  }

  const tempFile = path.join(os.tmpdir(), `geo-ai-${Date.now()}-${image.name}`);

  try {
    const bytes = await image.arrayBuffer();
    await writeFile(tempFile, new Uint8Array(bytes));

    const parsed = await runWorkerJob({
      image: tempFile,
      lat,
      lon,
      apiKey,
      confidence: Number.isFinite(confidence) ? confidence : 0.25,
      classVisibility: {
        vegetation: showVegetation,
        ruins: showRuins,
        structures: showStructures,
        boulders: showBoulders,
        others: showOthers,
      },
      useAiInsight,
    });

    if (!parsed.ok) {
      return NextResponse.json(
        {
          error: parsed.error || "Inference failed",
          details: parsed.traceback || "Worker returned an error",
        },
        { status: 500 },
      );
    }

    return NextResponse.json(parsed.data, { status: 200 });
  } catch (err) {
    return NextResponse.json(
      {
        error: "Unexpected API failure",
        details: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  } finally {
    try {
      await unlink(tempFile);
    } catch {
      // Ignore temp file cleanup errors.
    }

    // Recycle the Python worker to avoid stale imported modules during active development.
    shutdownWorker();
  }
}
