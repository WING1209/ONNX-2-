import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

const MODEL_URLS = {
  vocals:
    "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/main/vocals.onnx?download=true",
  accompaniment:
    "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/main/accompaniment.onnx?download=true",
};

const MODEL_CACHE_NAME = "wing-spleeter-onnx-cache-v1";
const STEM_KEYS = ["vocals", "accompaniment"];

// Spleeter 44.1kHz 標準 STFT パラメータ
const N_FFT = 4096;
const HOP_LENGTH = 1024;
const FREQ_BINS = N_FFT / 2 + 1; // 2049
const MODEL_BINS = 1024;

// メモリ負荷を抑えるため 6秒チャンク + 0.5秒オーバーラップに設定
const CHUNK_SAMPLES = 44100 * 6;
const OVERLAP_SAMPLES = Math.floor(44100 * 0.5);

function createPeriodicHannWindow(length) {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return win;
}

const HANN_WINDOW = createPeriodicHannWindow(N_FFT);

/**
 * ブラウザの Cache Storage からモデルバイナリを取得（初回のみDLして保存、2回目以降はキャッシュ利用）
 */
async function loadModelArrayBuffer(stemKey, url) {
  const cacheSupported = typeof caches !== "undefined";

  if (cacheSupported) {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cachedResponse = await cache.match(url);
    if (cachedResponse && cachedResponse.ok) {
      postMessage({
        type: "progress",
        message: `[${stemKey}] 保存済みキャッシュからモデル読込中...`,
      });
      return await cachedResponse.arrayBuffer();
    }

    postMessage({
      type: "progress",
      message: `[${stemKey}] モデルを初回ダウンロード＆ブラウザ保存中...`,
    });

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`${stemKey} モデルの取得に失敗しました (HTTP ${response.status})`);
    }

    // キャッシュ保存用にクローンして格納
    await cache.put(url, response.clone());
    return await response.arrayBuffer();
  }

  postMessage({
    type: "progress",
    message: `[${stemKey}] モデルをダウンロード中...`,
  });

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${stemKey} モデルの取得に失敗しました (HTTP ${response.status})`);
  }
  return await response.arrayBuffer();
}

/**
 * 1モデル分のセッションをロード
 */
async function createSingleSession(stemKey) {
  const url = MODEL_URLS[stemKey];
  const modelBuffer = await loadModelArrayBuffer(stemKey, url);

  postMessage({
    type: "progress",
    message: `[${stemKey}] AIセッションを展開中...`,
  });

  const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
    executionProviders: ["wasm"],
  });

  return session;
}

/**
 * セッションを明示的に解放
 */
async function releaseSession(session) {
  if (!session) return;
  try {
    if (typeof session.release === "function") {
      await session.release();
    }
  } catch (_) {
    // 解放時の軽微なエラーは無視
  }
}

/**
 * 基数2のインプレース FFT / IFFT (dir: 1 = FFT, -1 = IFFT)
 */
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

function buildSpleeterInputTensor(session, leftStft, rightStft) {
  const inputName = session.inputNames[0];
  const meta = session.inputMetadata?.[inputName];
  const dims = meta?.dimensions || [2, 1, 512, MODEL_BINS];

  const chDim = typeof dims[0] === "number" && dims[0] > 0 ? dims[0] : 2;
  const axis2 = typeof dims[2] === "number" && dims[2] > 0 ? dims[2] : 512;
  const freqBins = typeof dims[3] === "number" && dims[3] > 0 ? dims[3] : MODEL_BINS;

  const totalFrames = leftStft.numFrames;
  const numSplits =
    typeof dims[1] === "number" && dims[1] > 0
      ? dims[1]
      : Math.max(1, Math.ceil(totalFrames / axis2));

  const data = new Float32Array(chDim * numSplits * axis2 * freqBins);

  for (let c = 0; c < chDim; c++) {
    const srcRe = c === 0 ? leftStft.re : rightStft.re;
    const srcIm = c === 0 ? leftStft.im : rightStft.im;

    for (let sp = 0; sp < numSplits; sp++) {
      for (let t = 0; t < axis2; t++) {
        const frameIdx = sp * axis2 + t;
        const dstOffset = ((c * numSplits + sp) * axis2 + t) * freqBins;

        if (frameIdx < totalFrames) {
          const srcOffset = frameIdx * FREQ_BINS;
          const copyBins = Math.min(freqBins, FREQ_BINS);
          for (let b = 0; b < copyBins; b++) {
            const r = srcRe[srcOffset + b];
            const im = srcIm[srcOffset + b];
            data[dstOffset + b] = Math.hypot(r, im);
          }
        }
      }
    }
  }

  return {
    tensor: new ort.Tensor("float32", data, [chDim, numSplits, axis2, freqBins]),
    numSplits,
    axis2,
    freqBins,
  };
}

function reconChannelWithTensor(stftObj, chIdx, outTensor, metaInfo, outputLength) {
  const { numSplits, axis2, freqBins } = metaInfo;
  const dims = outTensor.dims || [];
  const chCountInOut = dims.length >= 1 && dims[0] >= 2 ? 2 : 1;
  const outData = outTensor.data;
  const totalFrames = stftObj.numFrames;

  const mRe = new Float32Array(stftObj.re.length);
  const mIm = new Float32Array(stftObj.im.length);

  for (let f = 0; f < totalFrames; f++) {
    const sp = Math.floor(f / axis2);
    const t = f % axis2;
    const base = f * FREQ_BINS;

    for (let b = 0; b < FREQ_BINS; b++) {
      const r = stftObj.re[base + b];
      const im = stftObj.im[base + b];
      const origMag = Math.hypot(r, im);

      let gain = 1.0;
      if (sp < numSplits && b < freqBins) {
        const actualCh = chCountInOut >= 2 ? chIdx : 0;
        const outIdx = ((actualCh * numSplits + sp) * axis2 + t) * freqBins + b;
        const predVal = outData[outIdx];
        if (typeof predVal === "number" && Number.isFinite(predVal)) {
          gain = origMag > 1e-6 ? Math.max(0, Math.min(2.0, predVal / origMag)) : 0;
        }
      }

      mRe[base + b] = r * gain;
      mIm[base + b] = im * gain;
    }
  }

  const audio = computeIstft(mRe, mIm, totalFrames, outputLength);
  return audio;
}

/**
 * 1つのモデル（vocals または accompaniment）について全チャンクを処理
 */
async function processSingleStem(stemKey, left, right, dstLeft, dstRight) {
  const totalLength = left.length;
  const step = Math.max(1, CHUNK_SAMPLES - OVERLAP_SAMPLES);
  const totalChunks = Math.max(1, Math.ceil(totalLength / step));

  let session = null;

  try {
    session = await createSingleSession(stemKey);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const start = chunkIdx * step;
      if (start >= totalLength) break;

      const end = Math.min(totalLength, start + CHUNK_SAMPLES);
      const pct = Math.round(((chunkIdx + 1) / totalChunks) * 100);

      postMessage({
        type: "progress",
        message: `[${stemKey}] 分離計算中... ${pct}% (${chunkIdx + 1}/${totalChunks})`,
      });

      const leftSlice = left.subarray(start, end);
      const rightSlice = right.subarray(start, end);

      let leftStft = computeStft(leftSlice);
      let rightStft = computeStft(rightSlice);

      const built = buildSpleeterInputTensor(session, leftStft, rightStft);
      const results = await session.run({ [inputName]: built.tensor });
      const outTensor = results[outputName];

      if (!outTensor) {
        built.tensor.dispose();
        throw new Error(`出力 "${outputName}" (${stemKey}) が取得できませんでした。`);
      }

      const outL = reconChannelWithTensor(leftStft, 0, outTensor, built, leftSlice.length);
      const outR = reconChannelWithTensor(rightStft, 1, outTensor, built, rightSlice.length);

      // チャンク用テンソル・STFT配列を即座に破棄
      built.tensor.dispose();
      outTensor.dispose();
      leftStft = null;
      rightStft = null;

      for (let i = 0; i < outL.length; i++) {
        const outIdx = start + i;
        if (outIdx >= totalLength) break;

        const overlapStart = outL.length - OVERLAP_SAMPLES;
        if (chunkIdx < totalChunks - 1 && OVERLAP_SAMPLES > 0 && i >= overlapStart) {
          const t = (i - overlapStart) / OVERLAP_SAMPLES;
          const fadeOut = 1 - t;
          dstLeft[outIdx] = dstLeft[outIdx] * fadeOut + outL[i] * t;
          dstRight[outIdx] = dstRight[outIdx] * fadeOut + outR[i] * t;
        } else if (i < step || chunkIdx === totalChunks - 1) {
          dstLeft[outIdx] = outL[i];
          dstRight[outIdx] = outR[i];
        }
      }
    }
  } finally {
    // 次のモデルに移る前に必ずセッションを解放
    await releaseSession(session);
    session = null;
  }
}

function clampBuffer(buf) {
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] > 1) buf[i] = 1;
    else if (buf[i] < -1) buf[i] = -1;
  }
}

self.onmessage = async (e) => {
  try {
    const { sampleRate, left, right } = e.data;
    const totalLength = left.length;

    const finalStems = {
      vocals: {
        left: new Float32Array(totalLength),
        right: new Float32Array(totalLength),
      },
      accompaniment: {
        left: new Float32Array(totalLength),
        right: new Float32Array(totalLength),
      },
    };

    // 1モデルずつ順番にロード・推論・解放して同時メモリ使用量を抑える
    for (let i = 0; i < STEM_KEYS.length; i++) {
      const key = STEM_KEYS[i];
      await processSingleStem(
        key,
        left,
        right,
        finalStems[key].left,
        finalStems[key].right
      );
      clampBuffer(finalStems[key].left);
      clampBuffer(finalStems[key].right);
    }

    postMessage({
      type: "progress",
      message: "2 パート分のWAVデータを転送中...",
    });

    const transferBuffers = [];
    for (const key of STEM_KEYS) {
      transferBuffers.push(finalStems[key].left.buffer, finalStems[key].right.buffer);
    }

    postMessage(
      {
        type: "done",
        sampleRate,
        stems: finalStems,
      },
      transferBuffers
    );
  } catch (err) {
    postMessage({
      type: "error",
      message: err?.message || String(err),
    });
  }
};
