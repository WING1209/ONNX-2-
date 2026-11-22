import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

// iOS Safari でのクラッシュ・WASMメモリ膨張を抑える設定
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.simd = false;

const ACCOMPANIMENT_MODEL_URL =
  "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/main/accompaniment.onnx?download=true";

const MODEL_CACHE_NAME = "wing-spleeter-accompaniment-cache-v1";

// Spleeter 44.1kHz 標準 STFT パラメータ
const N_FFT = 4096;
const HOP_LENGTH = 1024;
const FREQ_BINS = N_FFT / 2 + 1; // 2049
const MODEL_FRAMES = 512; // 1ブロックあたりのフレーム数 (約11.88秒分)
const MODEL_BINS = 1024;

// 1ブロックあたりのサンプル数（512フレーム分にぴったり合わせる）
const BLOCK_SAMPLES = MODEL_FRAMES * HOP_LENGTH; // 524,288 サンプル (約11.89秒)

function createPeriodicHannWindow(length) {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return win;
}

const HANN_WINDOW = createPeriodicHannWindow(N_FFT);

async function loadAccompanimentModelBuffer() {
  const cacheSupported = typeof caches !== "undefined";

  if (cacheSupported) {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cachedResponse = await cache.match(ACCOMPANIMENT_MODEL_URL);
    if (cachedResponse && cachedResponse.ok) {
      postMessage({
        type: "progress",
        message: "キャッシュから伴奏モデルを読込中...",
      });
      return await cachedResponse.arrayBuffer();
    }

    postMessage({
      type: "progress",
      message: "伴奏モデルを初回ダウンロード＆保存中...",
    });

    const response = await fetch(ACCOMPANIMENT_MODEL_URL);
    if (!response.ok) {
      throw new Error(`伴奏モデルの取得に失敗しました (HTTP ${response.status})`);
    }

    await cache.put(ACCOMPANIMENT_MODEL_URL, response.clone());
    return await response.arrayBuffer();
  }

  postMessage({
    type: "progress",
    message: "伴奏モデルをダウンロード中...",
  });

  const response = await fetch(ACCOMPANIMENT_MODEL_URL);
  if (!response.ok) {
    throw new Error(`伴奏モデルの取得に失敗しました (HTTP ${response.status})`);
  }
  return await response.arrayBuffer();
}

async function createAccompanimentSession() {
  let modelBuffer = await loadAccompanimentModelBuffer();

  postMessage({
    type: "progress",
    message: "AIセッションを展開中 (iOS軽量モード)...",
  });

  const modelBytes = new Uint8Array(modelBuffer);
  modelBuffer = null;

  return await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    enableMemPattern: false,
    enableCpuMemArena: false,
    graphOptimizationLevel: "basic",
  });
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

function buildSingleBlockTensor(session, leftStft, rightStft) {
  const inputName = session.inputNames[0];
  const meta = session.inputMetadata?.[inputName];
  const dims = meta?.dimensions || [2, 1, MODEL_FRAMES, MODEL_BINS];

  const chDim = typeof dims[0] === "number" && dims[0] > 0 ? dims[0] : 2;
  const numSplits = 1;
  const axis2 = typeof dims[2] === "number" && dims[2] > 0 ? dims[2] : MODEL_FRAMES;
  const freqBins = typeof dims[3] === "number" && dims[3] > 0 ? dims[3] : MODEL_BINS;

  const totalFrames = leftStft.numFrames;
  const data = new Float32Array(chDim * numSplits * axis2 * freqBins);

  for (let c = 0; c < chDim; c++) {
    const srcRe = c === 0 ? leftStft.re : rightStft.re;
    const srcIm = c === 0 ? leftStft.im : rightStft.im;

    const copyFrames = Math.min(axis2, totalFrames);
    for (let t = 0; t < copyFrames; t++) {
      const dstOffset = (c * axis2 + t) * freqBins;
      const srcOffset = t * FREQ_BINS;
      const copyBins = Math.min(freqBins, FREQ_BINS);
      for (let b = 0; b < copyBins; b++) {
        const r = srcRe[srcOffset + b];
        const im = srcIm[srcOffset + b];
        data[dstOffset + b] = Math.hypot(r, im);
      }
    }
  }

  return {
    tensor: new ort.Tensor("float32", data, [chDim, numSplits, axis2, freqBins]),
    axis2,
    freqBins,
  };
}

function reconChannelSingleBlock(stftObj, chIdx, outTensor, metaInfo, outputLength) {
  const { axis2, freqBins } = metaInfo;
  const dims = outTensor.dims || [];
  const chCountInOut = dims.length >= 1 && dims[0] >= 2 ? 2 : 1;
  const outData = outTensor.data;
  const totalFrames = stftObj.numFrames;

  const mRe = new Float32Array(stftObj.re.length);
  const mIm = new Float32Array(stftObj.im.length);

  for (let f = 0; f < totalFrames; f++) {
    const base = f * FREQ_BINS;

    for (let b = 0; b < FREQ_BINS; b++) {
      const r = stftObj.re[base + b];
      const im = stftObj.im[base + b];
      const origMag = Math.hypot(r, im);

      let gain = 1.0;
      if (f < axis2 && b < freqBins) {
        const actualCh = chCountInOut >= 2 ? chIdx : 0;
        const outIdx = (actualCh * axis2 + f) * freqBins + b;
        const predVal = outData[outIdx];
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

function clampBuffer(buf) {
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] > 1) buf[i] = 1;
    else if (buf[i] < -1) buf[i] = -1;
  }
}

// ブラウザにGC（メモリ解放）の余地を与えるための短い待機
function yieldForGC(ms = 80) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

self.onmessage = async (e) => {
  let session = null;

  try {
    const { sampleRate, left, right } = e.data;
    const totalLength = left.length;

    const dstLeft = new Float32Array(totalLength);
    const dstRight = new Float32Array(totalLength);

    const totalBlocks = Math.max(1, Math.ceil(totalLength / BLOCK_SAMPLES));

    session = await createAccompanimentSession();
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    for (let blockIdx = 0; blockIdx < totalBlocks; blockIdx++) {
      const start = blockIdx * BLOCK_SAMPLES;
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

      const built = buildSingleBlockTensor(session, leftStft, rightStft);
      const results = await session.run({ [inputName]: built.tensor });
      const outTensor = results[outputName];

      if (!outTensor) {
        built.tensor.dispose();
        throw new Error(`出力 "${outputName}" が取得できませんでした。`);
      }

      const outL = reconChannelSingleBlock(leftStft, 0, outTensor, built, leftSlice.length);
      const outR = reconChannelSingleBlock(rightStft, 1, outTensor, built, rightSlice.length);

      built.tensor.dispose();
      outTensor.dispose();
      leftStft = null;
      rightStft = null;

      dstLeft.set(outL, start);
      dstRight.set(outR, start);

      // 複数ブロックある場合は次の推論前に一瞬休ませてメモリ解放を促す
      if (blockIdx < totalBlocks - 1) {
        await yieldForGC(120);
      }
    }

    await releaseSession(session);
    session = null;

    clampBuffer(dstLeft);
    clampBuffer(dstRight);

    postMessage({
      type: "progress",
      message: "伴奏WAVデータを転送中...",
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
    postMessage({
      type: "error",
      message: err?.message || String(err),
    });
  }
};
