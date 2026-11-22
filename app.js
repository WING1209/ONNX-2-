const APP_VERSION = "Ver.1.3";
const TARGET_SAMPLE_RATE = 44100;
const MP3_BITRATE = 128;
const MP3_BLOCK_SIZE = 1152;
const LAMEJS_ESM_URL =
  "https://cdn.jsdelivr.net/npm/@breezystack/lamejs@1.2.7/dist/lame.es.js";

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const dlBtn = document.getElementById("dl-accompaniment");

let selectedFile = null;
let currentAudioBlob = null;
let currentBlobUrl = null;
let currentFileName = "accompaniment.mp3";
let lameEncoderClassPromise = null;

const worker = new Worker("worker.js?v=1.3", { type: "module" });

function isAllowedAudio(file) {
  if (!file) return false;
  const nameOk = /\.(mp3|wav)$/i.test(file.name);
  const typeOk = [
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
  ].includes(file.type.toLowerCase());
  return nameOk || typeOk;
}

function clearGeneratedAudio() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
  currentAudioBlob = null;
  if (dlBtn) {
    dlBtn.disabled = true;
    dlBtn.textContent = "MP3 保存";
  }
}

async function normalizeTo44100Stereo(decodedBuffer) {
  const numChannels = decodedBuffer.numberOfChannels;
  if (numChannels < 1) {
    throw new Error("有効な音声チャンネルが見つかりませんでした。");
  }

  const needsResample = decodedBuffer.sampleRate !== TARGET_SAMPLE_RATE;
  const needsChannelConvert = numChannels !== 2;

  if (!needsResample && !needsChannelConvert) {
    return decodedBuffer;
  }

  const targetLength = Math.ceil(decodedBuffer.duration * TARGET_SAMPLE_RATE);
  const offlineCtx = new OfflineAudioContext(2, targetLength, TARGET_SAMPLE_RATE);
  const inputForRender = offlineCtx.createBuffer(
    Math.min(numChannels, 2),
    decodedBuffer.length,
    decodedBuffer.sampleRate
  );

  if (numChannels === 1) {
    const mono = decodedBuffer.getChannelData(0);
    inputForRender.copyToChannel(mono, 0);
    inputForRender.copyToChannel(mono, 1);
  } else {
    inputForRender.copyToChannel(decodedBuffer.getChannelData(0), 0);
    inputForRender.copyToChannel(decodedBuffer.getChannelData(1), 1);
  }

  const source = offlineCtx.createBufferSource();
  source.buffer = inputForRender;
  source.connect(offlineCtx.destination);
  source.start(0);
  const rendered = await offlineCtx.startRendering();
  source.disconnect();
  return rendered;
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  clearGeneratedAudio();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式: MP3 / WAV (44100Hz・ステレオへ自動変換)";
    statusEl.textContent = "エラー: MP3 または WAVファイルのみ対応しています。";
    return;
  }

  selectedFile = file;
  const baseName = file.name.replace(/\.[^.]+$/, "") || "accompaniment";
  currentFileName = `${baseName}_accompaniment.mp3`;

  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  statusEl.textContent = `選択完了: ${file.name} (${APP_VERSION}) - 「伴奏抽出を実行！」を押してください。`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  clearGeneratedAudio();
  statusEl.textContent = "ステップ1/3: 音声データを読み込み中...";

  try {
    const arrayBuffer = await selectedFile.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextClass();
    let audioBuffer;

    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } finally {
      if (typeof audioCtx.close === "function") {
        await audioCtx.close().catch(() => {});
      }
    }

    const needsConvert =
      audioBuffer.sampleRate !== TARGET_SAMPLE_RATE ||
      audioBuffer.numberOfChannels !== 2;

    if (needsConvert) {
      statusEl.textContent = `ステップ1/3: 44100Hz・ステレオに変換中... (元: ${audioBuffer.sampleRate}Hz / ${audioBuffer.numberOfChannels}ch)`;
    }

    const normalizedBuffer = await normalizeTo44100Stereo(audioBuffer);
    const left = new Float32Array(normalizedBuffer.getChannelData(0));
    const right = new Float32Array(normalizedBuffer.getChannelData(1));

    statusEl.textContent = "ステップ2/3: モデル確認・伴奏抽出を開始中...";
    worker.postMessage(
      {
        sampleRate: normalizedBuffer.sampleRate,
        left,
        right,
      },
      [left.buffer, right.buffer]
    );
  } catch (err) {
    statusEl.textContent = `読み込み・変換エラー: ${err.message}`;
    runBtn.disabled = false;
  }
});

async function getMp3EncoderClass() {
  if (!lameEncoderClassPromise) {
    lameEncoderClassPromise = import(LAMEJS_ESM_URL)
      .then((mod) => {
        const Enc = mod?.Mp3Encoder || mod?.default?.Mp3Encoder;
        if (!Enc) {
          throw new Error("Mp3Encoder が見つかりませんでした。");
        }
        return Enc;
      })
      .catch((err) => {
        lameEncoderClassPromise = null;
        throw err;
      });
  }
  return await lameEncoderClassPromise;
}

function floatToInt16Chunk(floatArr, start, end, outInt16) {
  const len = end - start;
  for (let i = 0; i < len; i++) {
    const v = Math.max(-1, Math.min(1, floatArr[start + i]));
    outInt16[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return outInt16.subarray(0, len);
}

async function encodeMp3WithFallback(chL, chR, sampleRate, baseName) {
  try {
    const Mp3Encoder = await getMp3EncoderClass();
    const encoder = new Mp3Encoder(2, sampleRate, MP3_BITRATE);
    const mp3Chunks = [];
    const totalSamples = Math.min(chL.length, chR.length);

    const pcmL = new Int16Array(MP3_BLOCK_SIZE);
    const pcmR = new Int16Array(MP3_BLOCK_SIZE);

    for (let i = 0; i < totalSamples; i += MP3_BLOCK_SIZE) {
      const end = Math.min(totalSamples, i + MP3_BLOCK_SIZE);
      const subL = floatToInt16Chunk(chL, i, end, pcmL);
      const subR = floatToInt16Chunk(chR, i, end, pcmR);
      const encoded = encoder.encodeBuffer(subL, subR);
      if (encoded && encoded.length > 0) {
        mp3Chunks.push(new Uint8Array(encoded));
      }
    }

    const flushed = encoder.flush();
    if (flushed && flushed.length > 0) {
      mp3Chunks.push(new Uint8Array(flushed));
    }

    return {
      blob: new Blob(mp3Chunks, { type: "audio/mpeg" }),
      fileName: `${baseName}_accompaniment.mp3`,
      buttonLabel: "MP3 保存",
      isMp3: true,
    };
  } catch (_) {
    // MP3エンコーダ読込不可時のフォールバック（WAV生成）
    const numChannels = 2;
    const length = Math.min(chL.length, chR.length);
    const buffer = new ArrayBuffer(44 + length * numChannels * 2);
    const view = new DataView(buffer);

    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + length * numChannels * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, length * numChannels * 2, true);

    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sL = Math.max(-1, Math.min(1, chL[i]));
      const sR = Math.max(-1, Math.min(1, chR[i]));
      view.setInt16(offset, sL < 0 ? sL * 0x8000 : sL * 0x7fff, true);
      view.setInt16(offset + 2, sR < 0 ? sR * 0x8000 : sR * 0x7fff, true);
      offset += 4;
    }

    return {
      blob: new Blob([buffer], { type: "audio/wav" }),
      fileName: `${baseName}_accompaniment.wav`,
      buttonLabel: "WAV 保存",
      isMp3: false,
    };
  }
}

dlBtn.addEventListener("click", () => {
  if (!currentAudioBlob) return;

  try {
    if (!currentBlobUrl) {
      currentBlobUrl = URL.createObjectURL(currentAudioBlob);
    }

    const tempLink = document.createElement("a");
    tempLink.href = currentBlobUrl;
    tempLink.download = currentFileName;
    tempLink.style.display = "none";
    document.body.appendChild(tempLink);
    tempLink.click();
    document.body.removeChild(tempLink);

    statusEl.textContent = `ダウンロードを開始しました: ${currentFileName}`;

    setTimeout(() => {
      if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = null;
      }
    }, 3000);
  } catch (err) {
    statusEl.textContent = `ダウンロードエラー: ${err.message}`;
  }
});

worker.onmessage = async (e) => {
  const { type, message, accompaniment, sampleRate } = e.data;

  if (type === "progress") {
    statusEl.textContent = message;
  } else if (type === "done") {
    statusEl.textContent = `ステップ3/3: MP3 (${MP3_BITRATE}kbps) を生成中...`;

    try {
      if (accompaniment) {
        const baseName =
          selectedFile?.name?.replace(/\.[^.]+$/, "") || "accompaniment";
        const result = await encodeMp3WithFallback(
          accompaniment.left,
          accompaniment.right,
          sampleRate,
          baseName
        );
        currentAudioBlob = result.blob;
        currentFileName = result.fileName;
        accompaniment.left = null;
        accompaniment.right = null;

        if (dlBtn) {
          dlBtn.textContent = result.buttonLabel;
          dlBtn.disabled = false;
        }
      }

      statusEl.textContent = `完了! 「${dlBtn?.textContent || "保存"}」を押してダウンロードしてください。`;
    } catch (err) {
      statusEl.textContent = `音声変換エラー: ${err.message}`;
    } finally {
      runBtn.disabled = false;
    }
  } else if (type === "error") {
    statusEl.textContent = `分離エラー: ${message || "処理中にエラーが発生しました。"}`;
    runBtn.disabled = false;
  }
};

worker.onerror = (err) => {
  statusEl.textContent = `ワーカーエラー: ${err.message || "予期しないエラーが発生しました。"}`;
  runBtn.disabled = false;
};
