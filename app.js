import * as lamejs from "https://cdn.jsdelivr.net/npm/@breezystack/lamejs@1.2.7/dist/lame.es.js";

const APP_VERSION = "Ver.1.2";
const TARGET_SAMPLE_RATE = 44100;
const MP3_BITRATE = 128;
const MP3_BLOCK_SIZE = 1152;

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const dlBtn = document.getElementById("dl-accompaniment");

let selectedFile = null;
let currentMp3Blob = null;
let currentBlobUrl = null;
let currentFileName = "accompaniment.mp3";

const worker = new Worker("worker.js?v=1.2", { type: "module" });

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

function clearGeneratedMp3() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
  currentMp3Blob = null;
  if (dlBtn) {
    dlBtn.disabled = true;
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
  clearGeneratedMp3();

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
  clearGeneratedMp3();
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

function floatToInt16Chunk(floatArr, start, end, outInt16) {
  const len = end - start;
  for (let i = 0; i < len; i++) {
    const v = Math.max(-1, Math.min(1, floatArr[start + i]));
    outInt16[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return outInt16.subarray(0, len);
}

function encodeMp3(chL, chR, sampleRate, bitrate = MP3_BITRATE) {
  const Mp3Encoder = lamejs?.Mp3Encoder;
  if (!Mp3Encoder) {
    throw new Error("MP3エンコーダ（lamejs）の読み込みに失敗しました。");
  }

  const encoder = new Mp3Encoder(2, sampleRate, bitrate);
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

  return new Blob(mp3Chunks, { type: "audio/mpeg" });
}

dlBtn.addEventListener("click", () => {
  if (!currentMp3Blob) return;

  try {
    if (!currentBlobUrl) {
      currentBlobUrl = URL.createObjectURL(currentMp3Blob);
    }

    const tempLink = document.createElement("a");
    tempLink.href = currentBlobUrl;
    tempLink.download = currentFileName;
    tempLink.style.display = "none";
    document.body.appendChild(tempLink);
    tempLink.click();
    document.body.removeChild(tempLink);

    statusEl.textContent = `MP3ダウンロードを開始しました: ${currentFileName}`;

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

worker.onmessage = (e) => {
  const { type, message, accompaniment, sampleRate } = e.data;

  if (type === "progress") {
    statusEl.textContent = message;
  } else if (type === "done") {
    statusEl.textContent = `ステップ3/3: MP3 (${MP3_BITRATE}kbps) を生成中...`;

    try {
      if (accompaniment) {
        currentMp3Blob = encodeMp3(
          accompaniment.left,
          accompaniment.right,
          sampleRate,
          MP3_BITRATE
        );
        accompaniment.left = null;
        accompaniment.right = null;
        if (dlBtn) {
          dlBtn.disabled = false;
        }
      }

      statusEl.textContent = "完了! 「MP3 保存」を押してダウンロードしてください。";
    } catch (err) {
      statusEl.textContent = `MP3変換エラー: ${err.message}`;
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
