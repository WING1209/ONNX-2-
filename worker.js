import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.mjs";

// iOS Safari でのクラッシュ・WASMメモリ膨張を抑える設定
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.simd = false;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

const ACCOMPANIMENT_MODEL_URL =
  "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems-fp16/resolve/main/accompaniment.fp16.onnx?download=true";

const MODEL_CACHE_NAME = "wing-spleeter-accompaniment-fp16-cache-v1";

// Spleeter 44.1kHz 標準 STFT パラメータ
const N_FFT = 4096;
const HOP_LENGTH = 1024;
const FREQ_BINS = N_FFT / 2 + 1; // 2049

// モデル入力 x の仕様: [channels, splits, frames=512, bins=1024]
const MODEL_FRAMES = 512;
const MODEL_BINS = 1024;
const BLOCK_SAMPLES = MODEL_FRAMES * HOP_LENGTH; // 524,288 サンプル

// ブロック繋ぎ目のクロスフェード長（32フレーム）
const CROSSFADE_SAMPLES = 32 * HOP_LENGTH; // 32,768 サンプル
const STEP_SAMPLES = Math.max(HOP_LENGTH * 16, BLOCK_SAMPLES - CROSSFADE_SAMPLES);

// 4ブロックごとにセッションを解放・再生成して iOS の WASM ヒープ膨張を抑制
const SESSION_RECYCLE_BLOCKS = 4;

// IEEE 754 float32 <-> float16 (Uint16) 変換用バッファ
const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);

function floatToFloat16Bits(val) {
  f32View[0] = val;
  const bits = u32View[0];
  const sign = (bits >>> 16) & 0x8000;
  let exp = ((bits >>> 23) & 0xff) - 127 + 15;
  let frac = (bits >>> 13) & 0x03ff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    frac = (frac | 0x0400) >>> (1 - exp);
    return sign | frac;
  }
  if (exp >= 31) {
    return sign | 0x7c00;
  }
  return sign | (exp << 10) | frac;
}

function float16BitsToFloat(h) {
  const s = (h & 0x8000) << 16;
  const e = (h & 0x7c00) >>> 10;
  const f = h & 0x03ff;

  if (e === 0) {
    if (f === 0) {
      u32View[0] = s;
      return f32View[0];
    }
    let mant = f;
    let exp = -14;
    while ((mant & 0x0400) === 0) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x03ff;
    u32View[0] = s | ((exp + 127) << 23) | (mant << 13);
    return f32View[0];
  }
  if (e === 31) {
    u32View[0] = s | 0x7f800000 | (f << 13);
    return f32View[0];
  }
  u32View[0] = s | ((e - 15 + 127) << 23) | (f << 13);
  return f32View[0];
}

function createPeriodicHannWindow(length) {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return win;
}

const HANN_WINDOW = createPeriodicHannWindow(N_FFT);

function yieldForGC(ms = 90) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url, retries = 3) {
  let lastErr = null;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response;
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        postMessage({
          type: "progress",
          message: `通信を再試行中 (${attempt}/${retries})...`,
        });
        await yieldForGC(400 * attempt);
      }
    }
  }
  throw lastErr || new Error("ネットワーク取得に失敗しました。");
}

async function loadAccompanimentModelBytes() {
  const cacheSupported = typeof caches !== "undefined";

  if (cacheSupported) {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cachedResponse = await cache.match(ACCOMPANIMENT_MODEL_URL);
    if (cachedResponse && cachedResponse.ok) {
      postMessage({
        type: "progress",
        message: "キャッシュからFP16伴奏モデルを読込中...",
      });
      const buf = await cachedResponse.arrayBuffer();
      return new Uint8Array(buf);
    }

    postMessage({
      type: "progress",
      message: "FP16伴奏モデルを初回ダウンロード＆保存中...",
    });

    const response = await fetchWithRetry(ACCOMPANIMENT_MODEL_URL, 3);
    await cache.put(ACCOMPANIMENT_MODEL_URL, response.clone());
    const buf = await response.arrayBuffer();
    return new Uint8Array(buf);
  }

  postMessage({
    type: "progress",
    message: "FP16伴奏モデルをダウンロード中...",
  });

  const response = await fetchWithRetry(ACCOMPANIMENT_MODEL_URL, 3);
  const buf = await response.arrayBuffer();
  return new Uint8Array(buf);
}

async function createSessionFromBytes(modelBytes, retries = 3) {
  let lastErr = null;
  const cdnCandidates = [
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/",
    "https://unpkg.com/onnxruntime-web@1.19.2/dist/",
  ];

  for (let attempt = 1; attempt <= retries; attempt++) {
    ort.env.wasm.wasmPaths = cdnCandidates[(attempt - 1) % cdnCandidates.length];
    try {
      return await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        enableMemPattern: false,
        enableCpuMemArena: false,
        graphOptimizationLevel: "basic",
      });
    } catch (err) {
      lastErr = err;
      if (attempt < retries) {
        postMessage({
          type: "progress",
          message: `WASM初期化を再試行中 (${attempt}/${retries})...`,
        });
        await yieldForGC(500 * attempt);
      }
    }
  }

  throw lastErr || new Error("WASMセッションの初期化に失敗しました。");
}

async function releaseSession(session) {
  if (!session) return;
  try {
    if (typeof session.release === "function") {
      await session.release();
    }
  } catch (_) {}
}

function fftInPlace(re, im, dir) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tmpR = re[i];
      re[i] = re[j];
      re[j] = tmpR;
      const tmpI = im[i];
      im[i] = im[j];
      im[j] = tmpI;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (dir * -2 * Math.PI) / len;
    const wLenR = Math.cos(ang);
    const wLenI = Math.sin(ang);

    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;
      for (let k = 0; k < half; k++) {
        const uR = re[i + k];
        const uI = im[i + k];
        const vR = re[i + k + half] * wR - im[i + k + half] * wI;
        const vI = re[i + k + half] * wI + im[i + k + half] * wR;

        re[i + k] = uR + vR;
        im[i + k] = uI + vI;
        re[i + k + half] = uR - vR;
        im[i + k + half] = uI - vI;

        const nextWR = wR * wLenR - wI * wLenI;
        wI = wR * wLenI + wI * wLenR;
        wR = nextWR;
      }
    }
  }

  if (dir === -1) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

function computeStft(samples) {
  const numFrames = Math.max(1, Math.floor((samples.length - N_FFT) / HOP_LENGTH) + 1);
  const reOut = new Float32Array(numFrames * FREQ_BINS);
  const imOut = new Float32Array(numFrames * FREQ_BINS);

  const frameRe = new Float32Array(N_FFT);
  const frameIm = new Float32Array(N_FFT);

  for (let f = 0; f < numFrames; f++) {
    const start = f * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      const idx = start + i;
      const val = idx < samples.length ? samples[idx] : 0;
      frameRe[i] = val * HANN_WINDOW[i];
      frameIm[i] = 0;
    }

    fftInPlace(frameRe, frameIm, 1);

    const base = f * FREQ_BINS;
    for (let b = 0; b < FREQ_BINS; b++) {
      reOut[base + b] = frameRe[b];
      imOut[base + b] = frameIm[b];
    }
  }

  return { numFrames, re: reOut, im: imOut };
}

function computeIstft(reSpec, imSpec, numFrames, outputLength) {
  const out = new Float32Array(outputLength);
  const norm = new Float32Array(outputLength);
  const frameRe = new Float32Array(N_FFT);
  const frameIm = new Float32Array(N_FFT);

  for (let f = 0; f < numFrames; f++) {
    const base = f * FREQ_BINS;

    frameRe[0] = reSpec[base];
    frameIm[0] = 0;
    frameRe[N_FFT / 2] = reSpec[base + N_FFT / 2];
    frameIm[N_FFT / 2] = 0;

    for (let b = 1; b < N_FFT / 2; b++) {
      const r = reSpec[base + b];
      const im = imSpec[base + b];
      frameRe[b] = r;
      frameIm[b] = im;
      frameRe[N_FFT - b] = r;
      frameIm[N_FFT - b] = -im;
    }

    fftInPlace(frameRe, frameIm, -1);

    const start = f * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      const idx = start + i;
      if (idx < outputLength) {
        const w = HANN_WINDOW[i];
        out[idx] += frameRe[i] * w;
        norm[idx] += w * w;
      }
    }
  }

  for (let i = 0; i < outputLength; i++) {
    if (norm[i] > 1e-8) {
      out[i] /= norm[i];
    }
  }

  return out;
}

function buildFixed512Tensor(session, leftStft, rightStft) {
  const inputName = session.inputNames[0];
  const meta = session.inputMetadata?.[inputName];
  const dims = meta?.dimensions || [];
  const isFloat16Input = meta?.type === "float16";

  const chDim = typeof dims[0] === "number" && dims[0] > 0 ? dims[0] : 2;
  const numSplits = typeof dims[1] === "number" && dims[1] > 0 ? dims[1] : 1;
  const targetFrames =
    typeof dims[2] === "number" && dims[2] > 0 ? dims[2] : MODEL_FRAMES;
  const freqBins =
    typeof dims[3] === "number" && dims[3] > 0 ? dims[3] : MODEL_BINS;

  const totalFrames = leftStft.numFrames;
  const totalElements = chDim * numSplits * targetFrames * freqBins;
  const data = isFloat16Input
    ? new Uint16Array(totalElements)
    : new Float32Array(totalElements);

  for (let c = 0; c < chDim; c++) {
    const srcRe = c === 0 ? leftStft.re : rightStft.re;
    const srcIm = c === 0 ? leftStft.im : rightStft.im;

    for (let sp = 0; sp < numSplits; sp++) {
      const copyFrames = Math.min(targetFrames, Math.max(0, totalFrames - sp * targetFrames));
      for (let t = 0; t < copyFrames; t++) {
        const frameIdx = sp * targetFrames + t;
        const dstOffset = ((c * numSplits + sp) * targetFrames + t) * freqBins;
        const srcOffset = frameIdx * FREQ_BINS;
        const copyBins = Math.min(freqBins, FREQ_BINS);
        for (let b = 0; b < copyBins; b++) {
          const r = srcRe[srcOffset + b];
          const im = srcIm[srcOffset + b];
          const mag = Math.hypot(r, im);
          data[dstOffset + b] = isFloat16Input ? floatToFloat16Bits(mag) : mag;
        }
      }
    }
  }

  return {
    tensor: new ort.Tensor(
      isFloat16Input ? "float16" : "float32",
      data,
      [chDim, numSplits, targetFrames, freqBins]
    ),
    numSplits,
    targetFrames,
    freqBins,
  };
}

function readOutputValue(outTensor, outIdx) {
  const raw = outTensor.data[outIdx];
  if (outTensor.type === "float16" || outTensor.data instanceof Uint16Array) {
    return float16BitsToFloat(raw);
  }
  return raw;
}

function reconChannelFixed512(stftObj, chIdx, outTensor, specInfo, outputLength) {
  const { numSplits, targetFrames, freqBins } = specInfo;
  const dims = outTensor.dims || [];
  const chCountInOut = dims.length >= 1 && dims[0] >= 2 ? 2 : 1;
  const totalFrames = stftObj.numFrames;

  const mRe = new Float32Array(stftObj.re.length);
  const mIm = new Float32Array(stftObj.im.length);

  for (let f = 0; f < totalFrames; f++) {
    const sp = Math.floor(f / targetFrames);
    const t = f % targetFrames;
    const base = f * FREQ_BINS;

    for (let b = 0; b < FREQ_BINS; b++) {
      const r = stftObj.re[base + b];
      const im = stftObj.im[base + b];
      const origMag = Math.hypot(r, im);

      let gain = 1.0;
      if (sp < numSplits && b < freqBins) {
        const actualCh = chCountInOut >= 2 ? chIdx : 0;
        const outIdx = ((actualCh * numSplits + sp) * targetFrames + t) * freqBins + b;
        const predVal = readOutputValue(outTensor, outIdx);
        if (typeof predVal === "number" && Number.isFinite(predVal)) {
          gain = origMag > 1e-6 ? Math.max(0, Math.min(2.0, predVal / origMag)) : 0;
        }
      }

      mRe[base + b] = r * gain;
      mIm[base + b] = im * gain;
    }
  }

  return computeIstft(mRe, mIm, totalFrames, outputLength);
}

function writeBlockWithCrossfade(dstLeft, dstRight, outL, outR, start, totalLength, blockIdx, totalBlocks) {
  const len = Math.min(outL.length, outR.length);
  const hasPrevBlock = blockIdx > 0;
  const hasNextBlock = blockIdx < totalBlocks - 1;
  const fadeLen = Math.min(CROSSFADE_SAMPLES, Math.floor(len / 3));

  for (let i = 0; i < len; i++) {
    const outIdx = start + i;
    if (outIdx >= totalLength) break;

    let sampleL = outL[i];
    let sampleR = outR[i];

    if (hasPrevBlock && fadeLen > 0 && i < fadeLen) {
      const t = i / fadeLen;
      const wIn = 0.5 * (1 - Math.cos(Math.PI * t));
      const wPrev = 1 - wIn;
      sampleL = dstLeft[outIdx] * wPrev + sampleL * wIn;
      sampleR = dstRight[outIdx] * wPrev + sampleR * wIn;
    }

    if (hasNextBlock && fadeLen > 0 && i >= len - fadeLen) {
      const t = (i - (len - fadeLen)) / fadeLen;
      const wOut = 0.5 * (1 + Math.cos(Math.PI * t));
      dstLeft[outIdx] = sampleL * wOut;
      dstRight[outIdx] = sampleR * wOut;
    } else {
      dstLeft[outIdx] = sampleL;
      dstRight[outIdx] = sampleR;
    }
  }
}

function clampBuffer(buf) {
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] > 1) buf[i] = 1;
    else if (buf[i] < -1) buf[i] = -1;
  }
}

self.onmessage = async (e) => {
  let session = null;
  let modelBytes = null;

  try {
    const { sampleRate, left, right } = e.data;
    const totalLength = left.length;

    const dstLeft = new Float32Array(totalLength);
    const dstRight = new Float32Array(totalLength);

    modelBytes = await loadAccompanimentModelBytes();

    postMessage({
      type: "progress",
      message: "FP16 AIセッションを初期化中 (512フレーム仕様)...",
    });

    session = await createSessionFromBytes(modelBytes, 3);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const totalBlocks = Math.max(1, Math.ceil(totalLength / STEP_SAMPLES));

    for (let blockIdx = 0; blockIdx < totalBlocks; blockIdx++) {
      if (blockIdx > 0 && blockIdx % SESSION_RECYCLE_BLOCKS === 0) {
        postMessage({
          type: "progress",
          message: `メモリをリフレッシュ中... (${Math.round((blockIdx / totalBlocks) * 100)}%)`,
        });
        await releaseSession(session);
        session = null;
        await yieldForGC(140);
        session = await createSessionFromBytes(modelBytes, 3);
      }

      const start = blockIdx * STEP_SAMPLES;
      if (start >= totalLength) break;

      const end = Math.min(totalLength, start + BLOCK_SAMPLES);
      const pct = Math.round(((blockIdx + 1) / totalBlocks) * 100);

      postMessage({
        type: "progress",
        message: `伴奏を抽出中... ${pct}% (${blockIdx + 1}/${totalBlocks})`,
      });

      const leftSlice = left.subarray(start, end);
      const rightSlice = right.subarray(start, end);

      let leftStft = computeStft(leftSlice);
      let rightStft = computeStft(rightSlice);

      let built = buildFixed512Tensor(session, leftStft, rightStft);
      let results = await session.run({ [inputName]: built.tensor });
      const outTensor = results[outputName];

      if (!outTensor) {
        built.tensor.dispose();
        throw new Error(`出力 "${outputName}" が取得できませんでした。`);
      }

      const outL = reconChannelFixed512(leftStft, 0, outTensor, built, leftSlice.length);
      const outR = reconChannelFixed512(rightStft, 1, outTensor, built, rightSlice.length);

      built.tensor.dispose();
      outTensor.dispose();
      built = null;
      results = null;
      leftStft = null;
      rightStft = null;

      writeBlockWithCrossfade(
        dstLeft,
        dstRight,
        outL,
        outR,
        start,
        totalLength,
        blockIdx,
        totalBlocks
      );

      if (blockIdx < totalBlocks - 1) {
        await yieldForGC(90);
      }
    }

    await releaseSession(session);
    session = null;
    modelBytes = null;

    clampBuffer(dstLeft);
    clampBuffer(dstRight);

    postMessage({
      type: "progress",
      message: "伴奏データを転送中...",
    });

    postMessage(
      {
        type: "done",
        sampleRate,
        accompaniment: {
          left: dstLeft,
          right: dstRight,
        },
      },
      [dstLeft.buffer, dstRight.buffer]
    );
  } catch (err) {
    await releaseSession(session);
    modelBytes = null;
    postMessage({
      type: "error",
      message: err?.message || String(err),
    });
  }
};
