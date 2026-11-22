const APP_VERSION = "Ver.1.14";
const TARGET_SAMPLE_RATE = 44100;

// クラッシュを抑えるための出力設定・区間上限
const OUTPUT_SAMPLE_RATE = 22050;
const OUTPUT_CHANNELS = 1;
const DEFAULT_START_SEC = 0;
const DEFAULT_DURATION_SEC = 45;
const MAX_SLICE_SEC = 120;

// 1段階目のダウンロード処理開始時に不要なバックグラウンド／ページ参照を解放
const RELEASE_BACKGROUND_ON_DOWNLOAD = true;

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const dlBtn = document.getElementById("dl-accompaniment");
const autoSaveToggle = document.getElementById("autoSaveToggle");
const startSecInput = document.getElementById("startSecInput");
const durationSecInput = document.getElementById("durationSecInput");

let selectedFile = null;
let currentAudioBlob = null;
let currentBlobUrl = null;
let currentFileName = "accompaniment.wav";
let worker = null;
let autoSaveEnabled = true;

function updateAutoSaveUi() {
  if (!autoSaveToggle) return;
  autoSaveToggle.textContent = autoSaveEnabled ? "ON" : "OFF";
  autoSaveToggle.setAttribute("aria-pressed", String(autoSaveEnabled));
  autoSaveToggle.classList.toggle("is-on", autoSaveEnabled);
  autoSaveToggle.classList.toggle("is-off", !autoSaveEnabled);
}

if (autoSaveToggle) {
  updateAutoSaveUi();
  autoSaveToggle.addEventListener("click", () => {
    autoSaveEnabled = !autoSaveEnabled;
    updateAutoSaveUi();
    statusEl.textContent = autoSaveEnabled
      ? "自動保存を ON にしました（抽出完了時に保存を試行します）。"
      : "自動保存を OFF にしました（完了後に「音声 保存」を押してください）。";
  });
}

function getSliceRangeSec() {
  const rawStart = Number(startSecInput?.value ?? DEFAULT_START_SEC);
  const rawDur = Number(durationSecInput?.value ?? DEFAULT_DURATION_SEC);

  const startSec = Number.isFinite(rawStart) && rawStart >= 0 ? Math.floor(rawStart) : DEFAULT_START_SEC;
  let durationSec = Number.isFinite(rawDur) && rawDur > 0 ? Math.floor(rawDur) : DEFAULT_DURATION_SEC;
  if (durationSec > MAX_SLICE_SEC) {
    durationSec = MAX_SLICE_SEC;
    if (durationSecInput) durationSecInput.value = String(MAX_SLICE_SEC);
  }
  return { startSec, durationSec };
}

function isIosDevice() {
  const ua = navigator.userAgent || "";
  return /iPhone|iPad|iPod/i.test(ua);
}

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

function clearGeneratedAudio({ keepUrlBriefly = false } = {}) {
  if (currentBlobUrl && !keepUrlBriefly) {
    try {
      URL.revokeObjectURL(currentBlobUrl);
    } catch (_) {}
  }
  currentBlobUrl = null;
  currentAudioBlob = null;
  if (dlBtn) {
    dlBtn.disabled = true;
  }
}

function terminateWorkerSafely() {
  if (worker) {
    try {
      worker.terminate();
    } catch (_) {}
    worker = null;
  }
}

function releaseBackgroundMemoryOnDownload(urlToKeepBriefly = null) {
  terminateWorkerSafely();
  selectedFile = null;
  currentAudioBlob = null;

  if (fileInput) {
    try {
      fileInput.value = "";
    } catch (_) {}
  }

  if (dlBtn) {
    dlBtn.disabled = true;
  }

  if (urlToKeepBriefly) {
    setTimeout(() => {
      try {
        URL.revokeObjectURL(urlToKeepBriefly);
      } catch (_) {}
      if (currentBlobUrl === urlToKeepBriefly) {
        currentBlobUrl = null;
      }
    }, 20000);
  } else if (currentBlobUrl) {
    const oldUrl = currentBlobUrl;
    currentBlobUrl = null;
    setTimeout(() => {
      try {
        URL.revokeObjectURL(oldUrl);
      } catch (_) {}
    }, 20000);
  }
}

function yieldTick(ms = 25) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function extractNormalizedSlice(decodedBuffer, startSec, durationSec) {
  const numChannels = decodedBuffer.numberOfChannels;
  if (numChannels < 1) {
    throw new Error("有効な音声チャンネルが見つかりませんでした。");
  }

  const totalDuration = decodedBuffer.duration;
  const safeStart = Math.min(startSec, Math.max(0, totalDuration - 0.1));
  const actualDur = Math.max(0.1, Math.min(durationSec, totalDuration - safeStart));
  const targetLength = Math.max(1, Math.ceil(actualDur * TARGET_SAMPLE_RATE));

  const offlineCtx = new OfflineAudioContext(2, targetLength, TARGET_SAMPLE_RATE);

  const srcStartSample = Math.min(
    decodedBuffer.length,
    Math.max(0, Math.floor(safeStart * decodedBuffer.sampleRate))
  );
  const srcEndSample = Math.min(
    decodedBuffer.length,
    Math.max(srcStartSample + 1, Math.ceil((safeStart + actualDur) * decodedBuffer.sampleRate))
  );
  const sliceLen = srcEndSample - srcStartSample;

  const inputForRender = offlineCtx.createBuffer(
    Math.min(numChannels, 2),
    sliceLen,
    decodedBuffer.sampleRate
  );

  for (let c = 0; c < Math.min(numChannels, 2); c++) {
    const fullCh = decodedBuffer.getChannelData(c);
    inputForRender.copyToChannel(fullCh.subarray(srcStartSample, srcEndSample), c);
  }
  if (numChannels === 1) {
    const monoSlice = inputForRender.getChannelData(0);
    inputForRender.copyToChannel(monoSlice, 1);
  }

  const source = offlineCtx.createBufferSource();
  source.buffer = inputForRender;
  source.connect(offlineCtx.destination);
  source.start(0);
  const rendered = await offlineCtx.startRendering();
  source.disconnect();

  return {
    left: new Float32Array(rendered.getChannelData(0)),
    right: new Float32Array(rendered.getChannelData(1)),
    sampleRate: rendered.sampleRate,
    actualStartSec: Math.floor(safeStart),
    actualEndSec: Math.ceil(safeStart + actualDur),
  };
}

function buildOutputSamples(chL, chR, srcRate, dstRate, outChannels) {
  const ratio = srcRate / dstRate;
  const outLen = Math.max(1, Math.floor(chL.length / ratio));

  if (outChannels === 1) {
    const outMono = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcPos = i * ratio;
      const idx0 = Math.floor(srcPos);
      const idx1 = Math.min(chL.length - 1, idx0 + 1);
      const frac = srcPos - idx0;

      const sL = chL[idx0] * (1 - frac) + chL[idx1] * frac;
      const sR = chR[idx0] * (1 - frac) + chR[idx1] * frac;
      outMono[i] = 0.5 * (sL + sR);
    }
    return { channels: [outMono], rate: dstRate };
  }

  const outL = new Float32Array(outLen);
  const outR = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const idx0 = Math.floor(srcPos);
    const idx1 = Math.min(chL.length - 1, idx0 + 1);
    const frac = srcPos - idx0;

    outL[i] = chL[idx0] * (1 - frac) + chL[idx1] * frac;
    outR[i] = chR[idx0] * (1 - frac) + chR[idx1] * frac;
  }
  return { channels: [outL, outR], rate: dstRate };
}

async function triggerDownloadOrShare({ isAuto = false } = {}) {
  if (!currentAudioBlob || currentAudioBlob.size <= 44) {
    statusEl.textContent = "保存できる音声データがありません。";
    return false;
  }

  if (isIosDevice() && typeof navigator.share === "function") {
    try {
      const wavFile = new File([currentAudioBlob], currentFileName, {
        type: "audio/wav",
      });
      if (typeof navigator.canShare !== "function" || navigator.canShare({ files: [wavFile] })) {
        if (RELEASE_BACKGROUND_ON_DOWNLOAD) {
          terminateWorkerSafely();
          selectedFile = null;
          if (fileInput) {
            try {
              fileInput.value = "";
            } catch (_) {}
          }
        }
        await navigator.share({
          files: [wavFile],
          title: currentFileName,
        });
        statusEl.textContent = `共有メニューを開きました: ${currentFileName}`;
        if (RELEASE_BACKGROUND_ON_DOWNLOAD) {
          clearGeneratedAudio();
        }
        return true;
      }
    } catch (shareErr) {
      if (shareErr?.name === "AbortError") {
        statusEl.textContent = "保存をキャンセルしました。「音声 保存」から再試行できます。";
        return false;
      }
      if (isAuto) {
        statusEl.textContent = "自動共有はブロックされました。「音声 保存」を押して保存してください。";
        return false;
      }
    }
  }

  try {
    const urlToUse = currentBlobUrl || URL.createObjectURL(currentAudioBlob);
    currentBlobUrl = urlToUse;

    const tempLink = document.createElement("a");
    tempLink.href = urlToUse;
    tempLink.download = currentFileName;
    tempLink.rel = "noopener";
    tempLink.style.display = "none";
    document.body.appendChild(tempLink);
    tempLink.click();
    document.body.removeChild(tempLink);

    statusEl.textContent = isAuto
      ? `ダウンロード要求を送信しました: ${currentFileName}（ポップアップの「ダウンロード」を押してください）`
      : `ダウンロード要求を送信しました: ${currentFileName}`;

    if (RELEASE_BACKGROUND_ON_DOWNLOAD) {
      releaseBackgroundMemoryOnDownload(urlToUse);
    }

    return true;
  } catch (err) {
    statusEl.textContent = isAuto
      ? "自動保存がブロックされました。「音声 保存」を押して保存してください。"
      : `ダウンロードエラー: ${err.message}`;
    return false;
  }
}

function attachWorkerHandlers(w, baseName, startSec, endSec) {
  w.onmessage = async (e) => {
    const { type, message, accompaniment, sampleRate } = e.data;

    if (type === "progress") {
      statusEl.textContent = message;
    } else if (type === "done") {
      terminateWorkerSafely();
      statusEl.textContent = "ステップ3/3: 音声ファイルを生成中...";

      try {
        await yieldTick(30);

        if (accompaniment) {
          let chL = accompaniment.left;
          let chR = accompaniment.right;
          accompaniment.left = null;
          accompaniment.right = null;

          const builtOut = buildOutputSamples(
            chL,
            chR,
            sampleRate || TARGET_SAMPLE_RATE,
            OUTPUT_SAMPLE_RATE,
            OUTPUT_CHANNELS
          );
          chL = null;
          chR = null;

          currentAudioBlob = await encodeWavBlobChunked(
            builtOut.channels,
            builtOut.rate
          );
          builtOut.channels = null;
          currentFileName = `${baseName}_${startSec}-${endSec}s_accompaniment.wav`;

          if (dlBtn) {
            dlBtn.disabled = false;
          }
        }

        if (autoSaveEnabled && dlBtn && !dlBtn.disabled) {
          await triggerDownloadOrShare({ isAuto: true });
        } else {
          statusEl.textContent = "完了! 「音声 保存」を押して保存してください。";
        }
      } catch (err) {
        statusEl.textContent = `音声生成エラー: ${err.message}`;
      } finally {
        runBtn.disabled = false;
      }
    } else if (type === "error") {
      terminateWorkerSafely();
      statusEl.textContent = `分離エラー: ${message || "処理中にエラーが発生しました。"}`;
      runBtn.disabled = false;
    }
  };

  w.onerror = (err) => {
    terminateWorkerSafely();
    statusEl.textContent = `ワーカーエラー: ${err.message || "予期しないエラーが発生しました。"}`;
    runBtn.disabled = false;
  };
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  clearGeneratedAudio();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式: MP3 / WAV (指定区間を抽出)";
    statusEl.textContent = "エラー: MP3 または WAVファイルのみ対応しています。";
    return;
  }

  selectedFile = file;
  const baseName = file.name.replace(/\.[^.]+$/, "") || "accompaniment";
  const { startSec, durationSec } = getSliceRangeSec();
  currentFileName = `${baseName}_${startSec}-${startSec + durationSec}s_accompaniment.wav`;

  runBtn.disabled = false;
  const mb = (file.size / 1024 / 1024).toFixed(2);
  fileSub.textContent = `${file.name} (${mb} MB)`;
  statusEl.textContent = `選択完了: ${file.name} (${APP_VERSION}) - 「伴奏抽出を実行！」を押してください。`;
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.disabled = true;
  clearGeneratedAudio();
  terminateWorkerSafely();

  const fileRef = selectedFile;
  const baseName = fileRef.name.replace(/\.[^.]+$/, "") || "accompaniment";
  const { startSec, durationSec } = getSliceRangeSec();

  statusEl.textContent = `ステップ1/3: ${startSec}秒〜${startSec + durationSec}秒の区間を抽出中...`;

  try {
    let arrayBuffer = await fileRef.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextClass();
    let audioBuffer;

    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } finally {
      arrayBuffer = null;
      if (typeof audioCtx.close === "function") {
        await audioCtx.close().catch(() => {});
      }
    }

    const sliced = await extractNormalizedSlice(audioBuffer, startSec, durationSec);
    audioBuffer = null;

    worker = new Worker("worker.js?v=1.14-slice", { type: "module" });
    attachWorkerHandlers(worker, baseName, sliced.actualStartSec, sliced.actualEndSec);

    statusEl.textContent = `ステップ2/3: ${sliced.actualStartSec}〜${sliced.actualEndSec}秒の伴奏抽出を開始中...`;
    worker.postMessage(
      {
        sampleRate: sliced.sampleRate,
        left: sliced.left,
        right: sliced.right,
      },
      [sliced.left.buffer, sliced.right.buffer]
    );
  } catch (err) {
    terminateWorkerSafely();
    statusEl.textContent = `読み込み・切り出しエラー: ${err.message}`;
    runBtn.disabled = false;
  }
});

async function encodeWavBlobChunked(channels, sampleRate) {
  const numChannels = channels.length;
  const length = channels[0].length;
  const headerBuf = new ArrayBuffer(44);
  const headerView = new DataView(headerBuf);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      headerView.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  headerView.setUint32(4, 36 + length * numChannels * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  headerView.setUint32(16, 16, true);
  headerView.setUint16(20, 1, true);
  headerView.setUint16(22, numChannels, true);
  headerView.setUint32(24, sampleRate, true);
  headerView.setUint32(28, sampleRate * numChannels * 2, true);
  headerView.setUint16(32, numChannels * 2, true);
  headerView.setUint16(34, 16, true);
  writeStr(36, "data");
  headerView.setUint32(40, length * numChannels * 2, true);

  const parts = [headerBuf];
  const chunkSamples = Math.max(1, sampleRate * 2);

  for (let start = 0; start < length; start += chunkSamples) {
    const end = Math.min(length, start + chunkSamples);
    const segLen = end - start;
    const segBuf = new ArrayBuffer(segLen * numChannels * 2);
    const segView = new DataView(segBuf);

    let offset = 0;
    for (let i = 0; i < segLen; i++) {
      for (let c = 0; c < numChannels; c++) {
        const s = Math.max(-1, Math.min(1, channels[c][start + i]));
        segView.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
      }
    }

    parts.push(segBuf);
    await yieldTick(15);
  }

  return new Blob(parts, { type: "audio/wav" });
}

dlBtn.addEventListener("click", async () => {
  await triggerDownloadOrShare({ isAuto: false });
});
