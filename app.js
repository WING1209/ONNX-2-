const APP_VERSION = "Ver.1.1";
const TARGET_SAMPLE_RATE = 44100;

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const dlBtn = document.getElementById("dl-accompaniment");

let selectedFile = null;
let currentWavBlob = null;
let currentBlobUrl = null;
let currentFileName = "accompaniment.wav";

const worker = new Worker("worker.js?v=1.1", { type: "module" });

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

function clearGeneratedWav() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
  currentWavBlob = null;
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
  clearGeneratedWav();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式: MP3 / WAV (44100Hz・ステレオへ自動変換)";
    statusEl.textContent = "エラー: MP3 または WAVファイルのみ対応しています。";
    return;
  }

  selectedFile = file;
  const baseName = file.name.replace(/\.[^.]+$/, "") || "accompaniment";
  currentFileName = `${baseName}_accompaniment.wav`;

  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  statusEl.textContent = `選択完了: ${file.name} (${APP_VERSION}) - 「伴奏抽出を実行！」を押してください。`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  clearGeneratedWav();
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

function encodeWav(chL, chR, sampleRate) {
  const numChannels = 2;
  const length = chL.length;
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
  view.setUint16(20, 1, true); // PCM
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

  return new Blob([buffer], { type: "audio/wav" });
}

dlBtn.addEventListener("click", () => {
  if (!currentWavBlob) return;

  try {
    if (!currentBlobUrl) {
      currentBlobUrl = URL.createObjectURL(currentWavBlob);
    }

    const tempLink = document.createElement("a");
    tempLink.href = currentBlobUrl;
    tempLink.download = currentFileName;
    tempLink.style.display = "none";
    document.body.appendChild(tempLink);
    tempLink.click();
    document.body.removeChild(tempLink);

    statusEl.textContent = `ダウンロードを開始しました: ${currentFileName}`;

    // 保存開始後に少し待ってから Blob URL を解放し、メモリ保持を軽減
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
    statusEl.textContent = "ステップ3/3: 伴奏WAVファイルを生成中...";

    if (accompaniment) {
      currentWavBlob = encodeWav(accompaniment.left, accompaniment.right, sampleRate);
      accompaniment.left = null;
      accompaniment.right = null;
      if (dlBtn) {
        dlBtn.disabled = false;
      }
    }

    statusEl.textContent = "完了! 「WAV 保存」を押してダウンロードしてください。";
    runBtn.disabled = false;
  } else if (type === "error") {
    statusEl.textContent = `分離エラー: ${message || "処理中にエラーが発生しました。"}`;
    runBtn.disabled = false;
  }
};

worker.onerror = (err) => {
  statusEl.textContent = `ワーカーエラー: ${err.message || "予期しないエラーが発生しました。"}`;
  runBtn.disabled = false;
};
