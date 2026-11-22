import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs";

const VOCALS_MODEL_URL =
  "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/main/vocals.onnx?download=true";
const ACCOMPANIMENT_MODEL_URL =
  "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems/resolve/main/accompaniment.onnx?download=true";

const STEM_KEYS = ["vocals", "accompaniment"];

// Spleeter 44.1kHz 標準 STFT パラメータ
const N_FFT = 4096;
const HOP_LENGTH = 1024;
const FREQ_BINS = N_FFT / 2 + 1; // 2049
const MODEL_BINS = 1024; // U-Net 入力周波数ビン数 (~11kHz相当)

// チャンク処理用（約12秒ごとの分割＋オーバーラップで継ぎ目を軽減）
const CHUNK_SAMPLES = 44100 * 12;
const OVERLAP_SAMPLES = 44100 * 1;

let sessions = null;

function createPeriodicHannWindow(length) {
  const win = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / length));
  }
  return win;
}

const HANN_WINDOW = createPeriodicHannWindow(N_FFT);

async function initModels() {
  if (sessions) return sessions;

  postMessage({
    type: "progress",
    message: "2ステム用AIモデル (vocals / accompaniment) を読み込み中...",
  });

  const [vocalsSession, accompanimentSession] = await Promise.all([
    ort.InferenceSession.create(VOCALS_MODEL_URL, {
      executionProviders: ["wasm"],
    }),
    ort.InferenceSession.create(ACCOMPANIMENT_MODEL_URL, {
      executionProviders: ["wasm"],
    }),
  ]);

  sessions = {
    vocals: vocalsSession,
    accompaniment: accompanimentSession,
  };

  return sessions;
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
    const ang = ((dir * -2 * Math.PI) / len);
    const wLenR = Math.cos(ang);
    const wLenI = Math.sin(ang);

    for (let i = 0; i < n; i += len) {
      let wR = 1;
      let wI = 0;
      for (let k = 0; k < half; k++) {
        const uR = re[i + k];
        const uI = im[i + k];
        const vR = re[i + k + half] * wR - im[i + k + half] * wI;
        const vI = re[i + k + half] * wR ? re[i + k + half] * wI + im[i + k + half] * wR : re[i + k + half] * wI + im[i + k + half] * wR;

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

/**
 * 1チャンネル波形から STFT を計算
 * 戻り値: { numFrames, re: Float32Array[numFrames * FREQ_BINS], im: Float32Array[numFrames * FREQ_BINS] }
 */
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

/**
 * 複素 STFT から iSTFT で波形へ復元
 */
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

/**
 * モデルの入力次元に合わせて振幅テンソルを組み立てる
 * 代表的な Spleeter ONNX 入力形状: [2, num_splits, 512, 1024]
 */
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
        const dstOffset = (((c * numSplits + sp) * axis2 + t) * freqBins);

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

/**
 * モデル出力（振幅またはマスク）を元の複素 STFT に適用して復元する
 */
function applyModelOutputToStft(outTensor, stft, metaInfo) {
  const { numSplits, axis2, freqBins } = metaInfo;
  const outData = outTensor.data;
  const totalFrames = stft.numFrames;

  const maskedRe = new Float32Array(stft.re.length);
  const maskedIm = new Float32Array(stft.im.length);

  for (let f = 0; f < totalFrames; f++) {
    const sp = Math.floor(f / axis2);
    const t = f % axis2;
    const base = f * FREQ_BINS;

    // 11kHz以上の高域帯域はフレーム平均ゲインで補完
    let bandMeanGain = 1.0;
    if (sp < numSplits) {
      let sumRatio = 0;
      let count = 0;
      const checkBins = Math.min(freqBins, FREQ_BINS);
      for (let b = 0; b < checkBins; b++) {
        const r = stft.re[base + b];
        const im = stft.im[base + b];
        const origMag = Math.hypot(r, im);
        if (origMag > 1e-6) {
          const outIdx = (((0 * numSplits + sp) * axis2 + t) * freqBins) + b;
          const predVal = outData[outIdx] ?? origMag;
          sumRatio += Math.max(0, Math.min(1.5, predVal / origMag));
          count++;
        }
      }
      if (count > 0) {
        bandMeanGain = sumRatio / count;
      }
    }

    for (let b = 0; b < FREQ_BINS; b++) {
      const r = stft.re[base + b];
      const im = stft.im[base + b];
      const origMag = Math.hypot(r, im);

      let gain = bandMeanGain;
      if (sp < numSplits && b < freqBins) {
        const outIdx = (((0 * numSplits + sp) * axis2 + t) * freqBins) + b;
        const predVal = outData[outIdx];
        if (typeof predVal === "number" && Number.isFinite(predVal)) {
          // predVal が推定振幅に近い場合は比率を、マスク値(0..1)に近い場合はそのまま適用
          gain = origMag > 1e-6 ? Math.max(0, Math.min(2.0, predVal / origMag)) : 0;
        }
      }

      maskedRe[base + b] = r * gain;
      maskedIm[base + b] = im * gain;
    }
  }

  return maskedRe, maskedIm;
}

/**
 * 1チャンク分のステレオ音声を 2ステム (vocals / accompaniment) に分離
 */
async function separateChunk(loadedSessions, leftChunk, rightChunk) {
  const leftStft = computeStft(leftChunk);
  const rightStft = computeStft(rightChunk);

  const chunkResult = {
    vocals: {
      left: new Float32Array(leftChunk.length),
      right: new Float32Array(rightChunk.length),
    },
    accompaniment: {
      left: new Float32Array(leftChunk.length),
      right: new Float32Array(rightChunk.length),
    },
  };

  for (const key of STEM_KEYS) {
    const session = loadedSessions[key];
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const built = buildSpleeterInputTensor(session, leftStft, rightStft);
    const results = await session.run({ [inputName]: built.tensor });
    const outTensor = results[outputName];

    if (!outTensor) {
      built.tensor.dispose();
      throw new Error(`出力 "${outputName}" (${key}) が取得できませんでした。`);
    }

    // 左右チャンネルそれぞれの出力を復元
    const dims = outTensor.dims || [];
    const chCountInOut = dims.length >= 1 && dims[0] >= 2 ? 2 : 1;

    const reconChannel = (stftObj, chIdx) => {
      const spCount = built.numSplits;
      const tCount = built.axis2;
      const fCount = built.freqBins;
      const outData = outTensor.data;
      const totalFrames = stftObj.numFrames;

      const mRe = new Float32Array(stftObj.re.length);
      const mIm = new Float32Array(stftObj.im.length);

      for (let f = 0; f < totalFrames; f++) {
        const sp = Math.floor(f / tCount);
        const t = f % tCount;
        const base = f * FREQ_BINS;

        for (let b = 0; b < FREQ_BINS; b++) {
          const r = stftObj.re[base + b];
          const im = stftObj.im[base + b];
          const origMag = Math.hypot(r, im);

          let gain = 1.0;
          if (sp < spCount && b < fCount) {
            const actualCh = chCountInOut >= 2 ? chIdx : 0;
            const outIdx = (((actualCh * spCount + sp) * tCount + t) * fCount) + b;
            const predVal = outData[outIdx];
            if (typeof predVal === "number" && Number.isFinite(predVal)) {
              gain = origMag > 1e-6 ? Math.max(0, Math.min(2.0, predVal / origMag)) : 0;
            }
          }

          mRe[base + b] = r * gain;
          mIm[base + b] = im * gain;
        }
      }

      return computeIstft(mRe, mIm, totalFrames, leftChunk.length);
    };

    chunkResult[key].left = reconChannel(leftStft, 0);
    chunkResult[key].right = reconChannel(rightStft, 1);

    built.tensor.dispose();
    outTensor.dispose();
  }

  return chunkResult;
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

    const loadedSessions = await initModels();

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

    const step = Math.max(1, CHUNK_SAMPLES - OVERLAP_SAMPLES);
    const totalChunks = Math.max(1, Math.ceil(totalLength / step));

    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const start = chunkIdx * step;
      if (start >= totalLength) break;

      const end = Math.min(totalLength, start + CHUNK_SAMPLES);
      const pct = Math.round((chunkIdx / totalChunks) * 100);

      postMessage({
        type: "progress",
        message: `2 パート分離を計算中... (${pct}%)`,
      });

      const leftSlice = left.subarray(start, end);
      const rightSlice = right.subarray(start, end);

      const chunkOut = await separateChunk(loadedSessions, leftSlice, rightSlice);

      for (const key of STEM_KEYS) {
        const dstL = finalStems[key].left;
        const dstR = finalStems[key].right;
        const srcL = chunkOut[key].left;
        const srcR = chunkOut[key].right;

        for (let i = 0; i < srcL.length; i++) {
          const outIdx = start + i;
          if (outIdx >= totalLength) break;

          // オーバーラップ領域はクロスフェードで滑らかに接続
          const overlapStart = srcL.length - OVERLAP_SAMPLES;
          if (chunkIdx < totalChunks - 1 && OVERLAP_SAMPLES > 0 && i >= overlapStart) {
            const t = (i - overlapStart) / OVERLAP_SAMPLES;
            const fadeOut = 1 - t;
            dstL[outIdx] = dstL[outIdx] * fadeOut + srcL[i] * t;
            dstR[outIdx] = dstR[outIdx] * fadeOut + srcR[i] * t;
          } else if (outIdx < start + step || chunkIdx === totalChunks - 1) {
            if (i < step || chunkIdx === totalChunks - 1) {
              dstL[outIdx] = srcL[i];
              dstR[outIdx] = srcR[i];
            }
          }
        }
      }
    }

    for (const key of STEM_KEYS) {
      clampBuffer(finalStems[key].left);
      clampBuffer(finalStems[key].right);
    }

    postMessage({
      type: "progress",
      message: "2 パート分のWAVファイルを生成中...",
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