const APP_VERSION = "Ver.1.14";
const TARGET_SAMPLE_RATE = 44100;

// クラッシュを抑えるための出力設定・区間上限
const OUTPUT_SAMPLE_RATE = 22050;
const OUTPUT_CHANNELS = 1;
const DEFAULT_START_SEC = 0;
const DEFAULT_DURATION_SEC = 45;
const MAX_SLICE_SEC = 120;

// ダウンロード処理開始時に不要なバックグラウンド／ページ参照を解放
const RELEASE_BACKGROUND_ON_DOWNLOAD = true;

const fileInput = document.getElementById("audioFile");
const fileSub = document.getElementById("fileSub");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const dlBtn = document.getElementById("dl-accompaniment");
const autoSaveToggle = document.getElementById("autoSaveToggle");
const startSecInput = document.getElementById("startSecInput");
const durationSecInput = document.getElementById("durationSecInput");

const mergeFilesInput = document.getElementById("mergeFilesInput");
const mergePickSub = document.getElementById("mergePickSub");
const mergeFileList = document.getElementById("mergeFileList");
const mergeBtn = document.getElementById("mergeBtn");

let selectedFile = null;
let selectedMergeFiles = [];
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
      ? "自動保存を ON にしました（完了時に保存を試行します）。"
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

function buildAutoSegments(totalDurationSec, startSec, stepDurationSec) {
  const segments = [];
  if (totalDurationSec <= 0) return segments;

  let curStart = Math.min(startSec, Math.max(0, totalDurationSec - 0.1));
  while (curStart < totalDurationSec - 0.05) {
    const curEnd = Math.min(totalDurationSec, curStart + stepDurationSec);
    const actualLen = curEnd - curStart;
    if (actualLen >= 0.05) {
      segments.push({
        startSec: curStart,
        endSec: curEnd,
        labelStart: Math.floor(curStart),
        labelEnd: Math.ceil(curEnd),
      });
    }
    curStart = curEnd;
  }
  return segments;
}

function extractStartSecFromName(fileName) {
  const match = fileName.match(/(?:^|[_-])(\d+)-(\d+)s/i);
  if (match) {
    return Number(match[1]);
  }
  return Number.MAX_SAFE_INTEGER;
}

function sortMergeFiles(files) {
  return [...files].sort((a, b) => {
    const startA = extractStartSecFromName(a.name);
    const startB = extractStartSecFromName(b.name);
    if (startA !== startB) return startA - startB;
    return a.name.localeCompare(b.name, "ja");
  });
}

function renderMergeFileList(files) {
  if (!mergeFileList) return;
  mergeFileList.innerHTML = "";
  files.forEach((f, idx) => {
    const li = document.createElement("li");
    li.textContent = `${idx + 1}. ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
    mergeFileList.appendChild(li);
  });
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

function clearGeneratedAudio({ keepUrlBriefly = false, keepInputFile = false } = {}) {
  if (currentBlobUrl && !keepUrlBriefly) {
    try {
      URL.revokeObjectURL(currentBlobUrl);
    } catch (_) {}
  }
  currentBlobUrl = null;
  currentAudioBlob = null;
  if (!keepInputFile && dlBtn) {
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

function releaseSegmentMemoryAfterSave(urlToKeepBriefly = null, isFinalSegment = false) {
  terminateWorkerSafely();
  currentAudioBlob = null;

  if (isFinalSegment) {
    selectedFile = null;
    if (fileInput) {
      try {
        fileInput.value = "";
      } catch (_) {}
    }
    if (dlBtn) {
      dlBtn.disabled = true;
    }
  }

  if (urlToKeepBriefly) {
    setTimeout(() => {
      try {
        URL.revokeObjectURL(urlToKeepBriefly);
      } catch (_) {}
      if (currentBlobUrl === urlToKeepBriefly) {
        currentBlobUrl = null;
      }
    }, 15000);
  } else if (currentBlobUrl) {
    const oldUrl = currentBlobUrl;
    currentBlobUrl = null;
    setTimeout(() => {
      try {
        URL.revokeObjectURL(oldUrl);
      } catch (_) {}
    }, 15000);
  }
}

function yieldTick(ms = 30) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function decodeFileToMonoResampled(file, targetRate) {
  let arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContextClass();
  let decodedBuffer = null;

  try {
    decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  } finally {
    arrayBuffer = null;
    if (typeof audioCtx.close === "function") {
      await audioCtx.close().catch(() => {});
    }
  }

  const length = Math.max(1, Math.ceil(decodedBuffer.duration * targetRate));
  const offlineCtx = new OfflineAudioContext(1, length, targetRate);
  const src = offlineCtx.createBufferSource();
  src.buffer = decodedBuffer;
  src.connect(offlineCtx.destination);
  src.start(0);
  const rendered = await offlineCtx.startRendering();
  src.disconnect();
  decodedBuffer = null;

  return new Float32Array(rendered.getChannelData(0));
}

async function extractNormalizedSlice(decodedBuffer, startSec, endSec) {
  const numChannels = decodedBuffer.numberOfChannels;
  if (numChannels < 1) {
    throw new Error("有効な音声チャンネルが見つかりませんでした。");
  }

  const totalDuration = decodedBuffer.duration;
  const safeStart = Math.min(startSec, Math.max(0, totalDuration - 0.05));
  const safeEnd = Math.min(totalDuration, Math.max(safeStart + 0.05, endSec));
  const actualDur = safeEnd - safeStart;
  const targetLength = Math.max(1, Math.ceil(actualDur * TARGET_SAMPLE_RATE));

  const offlineCtx = new OfflineAudioContext(2, targetLength, TARGET_SAMPLE_RATE);

  const srcStartSample = Math.min(
    decodedBuffer.length,
    Math.max(0, Math.floor(safeStart * decodedBuffer.sampleRate))
  );
  const srcEndSample = Math.min(
    decodedBuffer.length,
    Math.max(srcStartSample + 1, Math.ceil(safeEnd * decodedBuffer.sampleRate))
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

async function triggerDownloadOrShare({ isAuto = false, isFinalSegment = false } = {}) {
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
        await navigator.share({
          files: [wavFile],
          title: currentFileName,
        });
        statusEl.textContent = `共有メニューを開きました: ${currentFileName}`;
        if (RELEASE_BACKGROUND_ON_DOWNLOAD) {
          releaseSegmentMemoryAfterSave(null, isFinalSegment);
        }
        return true;
      }
    } catch (shareErr) {
      if (shareErr?.name === "AbortError") {
        statusEl.textContent = `保存をキャンセルしました: ${currentFileName}（「音声 保存」から再試行できます）`;
        return false;
      }
      if (isAuto) {
        statusEl.textContent = `自動共有はブロックされました: ${currentFileName}（「音声 保存」を押してください）`;
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
      ? `ダウンロード要求を送信しました: ${currentFileName}`
      : `ダウンロード要求を送信しました: ${currentFileName}`;

    if (RELEASE_BACKGROUND_ON_DOWNLOAD) {
      releaseSegmentMemoryAfterSave(urlToKeepBriefly, isFinalSegment);
    }

    return true;
  } catch (err) {
    statusEl.textContent = isAuto
      ? `自動保存がブロックされました: ${currentFileName}（「音声 保存」を押してください）`
      : `ダウンロードエラー: ${err.message}`;
    return false;
  }
}

function runWorkerForSegment(sampleRate, left, right, segIndex, totalSegments, segLabel) {
  return new Promise((resolve, reject) => {
    terminateWorkerSafely();
    const w = new Worker("worker.js?v=1.14-fp16-merge", { type: "module" });
    worker = w;

    w.onmessage = (e) => {
      const { type, message, accompaniment, sampleRate: outRate } = e.data;
      if (type === "progress") {
        statusEl.textContent = `[区間 ${segIndex + 1}/${totalSegments} (${segLabel})] ${message}`;
      } else if (type === "done") {
        terminateWorkerSafely();
        resolve({
          left: accompaniment?.left || new Float32Array(0),
          right: accompaniment?.right || new Float32Array(0),
          sampleRate: outRate || sampleRate,
        });
      } else if (type === "error") {
        terminateWorkerSafely();
        reject(new Error(message || "区間分離中にエラーが発生しました。"));
      }
    };

    w.onerror = (err) => {
      terminateWorkerSafely();
      reject(new Error(err?.message || "ワーカーエラーが発生しました。"));
    };

    w.postMessage(
      {
        sampleRate,
        left,
        right,
      },
      [left.buffer, right.buffer]
    );
  });
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  clearGeneratedAudio();

  if (!isAllowedAudio(file)) {
    selectedFile = null;
    runBtn.disabled = true;
    fileSub.textContent = "対応形式: MP3 / WAV (曲終了まで自動分割)";
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
  if (mergeBtn) mergeBtn.disabled = true;
  clearGeneratedAudio();
  terminateWorkerSafely();

  const fileRef = selectedFile;
  const baseName = fileRef.name.replace(/\.[^.]+$/, "") || "accompaniment";
  const { startSec, durationSec } = getSliceRangeSec();

  statusEl.textContent = "ステップ1/3: 音源ファイルをデコード中...";

  let audioBuffer = null;

  try {
    let arrayBuffer = await fileRef.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextClass();

    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } finally {
      arrayBuffer = null;
      if (typeof audioCtx.close === "function") {
        await audioCtx.close().catch(() => {});
      }
    }

    const segments = buildAutoSegments(audioBuffer.duration, startSec, durationSec);
    if (segments.length === 0) {
      throw new Error("指定された開始位置が音源の長さを超えています。");
    }

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const segLabel = `${seg.labelStart}-${seg.labelEnd}s`;
      const isFinalSegment = i === segments.length - 1;

      statusEl.textContent = `[区間 ${i + 1}/${segments.length} (${segLabel})] 音声を切り出し中...`;
      let sliced = await extractNormalizedSlice(audioBuffer, seg.startSec, seg.endSec);

      let sepResult = await runWorkerForSegment(
        sliced.sampleRate,
        sliced.left,
        sliced.right,
        i,
        segments.length,
        segLabel
      );

      sliced.left = null;
      sliced.right = null;
      sliced = null;

      statusEl.textContent = `[区間 ${i + 1}/${segments.length} (${segLabel})] 音声ファイルを生成中...`;
      await yieldTick(25);

      let chL = sepResult.left;
      let chR = sepResult.right;
      const outRate = sepResult.sampleRate;
      sepResult = null;

      const builtOut = buildOutputSamples(
        chL,
        chR,
        outRate,
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
      currentFileName = `${baseName}_${seg.labelStart}-${seg.labelEnd}s_accompaniment.wav`;

      if (dlBtn) {
        dlBtn.disabled = false;
      }

      if (autoSaveEnabled) {
        await triggerDownloadOrShare({ isAuto: true, isFinalSegment });
      } else {
        statusEl.textContent = isFinalSegment
          ? `完了 (${segLabel})! 「音声 保存」を押して保存してください。`
          : `区間 ${i + 1}/${segments.length} (${segLabel}) 完了! 「音声 保存」を押してから次の区間へ進みます...`;
      }

      if (!isFinalSegment) {
        await yieldTick(80);
      }
    }

    audioBuffer = null;
    if (!autoSaveEnabled) {
      statusEl.textContent = `全 ${segments.length} 区間の抽出が完了しました！`;
    }
  } catch (err) {
    audioBuffer = null;
    terminateWorkerSafely();
    statusEl.textContent = `処理エラー: ${err.message}`;
  } finally {
    audioBuffer = null;
    runBtn.disabled = !selectedFile;
    if (mergeBtn) mergeBtn.disabled = selectedMergeFiles.length < 2;
  }
});

if (mergeFilesInput) {
  mergeFilesInput.addEventListener("change", (e) => {
    const rawFiles = Array.from(e.target.files || []).filter(isAllowedAudio);
    clearGeneratedAudio();

    if (rawFiles.length < 2) {
      selectedMergeFiles = [];
      renderMergeFileList([]);
      if (mergePickSub) {
        mergePickSub.textContent = "0-45s / 45-90s などのファイルをまとめて選択（秒数順に自動整列）";
      }
      if (mergeBtn) mergeBtn.disabled = true;
      statusEl.textContent = "統合するには細切れファイルを2つ以上選択してください。";
      return;
    }

    selectedMergeFiles = sortMergeFiles(rawFiles);
    renderMergeFileList(selectedMergeFiles);
    if (mergePickSub) {
      mergePickSub.textContent = `${selectedMergeFiles.length} 個のファイルをロード済み`;
    }
    if (mergeBtn) mergeBtn.disabled = false;
    statusEl.textContent = `${selectedMergeFiles.length} 個の細切れファイルを読み込みました。「選択したファイルを1つに統合！」を押してください。`;
  });
}

if (mergeBtn) {
  mergeBtn.addEventListener("click", async () => {
    if (selectedMergeFiles.length < 2) return;

    mergeBtn.disabled = true;
    runBtn.disabled = true;
    clearGeneratedAudio();
    terminateWorkerSafely();

    const partsMono = [];

    try {
      let totalSamples = 0;

      for (let i = 0; i < selectedMergeFiles.length; i++) {
        const f = selectedMergeFiles[i];
        statusEl.textContent = `統合準備中 (${i + 1}/${selectedMergeFiles.length}): ${f.name} をデコード中...`;
        const monoSamples = await decodeFileToMonoResampled(f, OUTPUT_SAMPLE_RATE);
        partsMono.push(monoSamples);
        totalSamples += monoSamples.length;
        await yieldTick(25);
      }

      statusEl.textContent = "細切れデータを1本の音声に結合中...";
      await yieldTick(25);

      let mergedMono = new Float32Array(Math.max(1, totalSamples));
      let writePos = 0;
      for (let i = 0; i < partsMono.length; i++) {
        const seg = partsMono[i];
        mergedMono.set(seg, writePos);
        writePos += seg.length;
        partsMono[i] = null;
      }
      partsMono.length = 0;

      statusEl.textContent = "統合WAVファイルを生成中...";
      currentAudioBlob = await encodeWavBlobChunked([mergedMono], OUTPUT_SAMPLE_RATE);
      mergedMono = null;

      const firstNameBase =
        selectedMergeFiles[0].name.replace(/(_\d+-\d+s.*)?\.[^.]+$/i, "") || "merged";
      currentFileName = `${firstNameBase}_merged_accompaniment.wav`;

      if (dlBtn) {
        dlBtn.disabled = false;
      }

      if (autoSaveEnabled) {
        await triggerDownloadOrShare({ isAuto: true, isFinalSegment: false });
      } else {
        statusEl.textContent = `統合完了 (${currentFileName})! 「音声 保存」を押して保存してください。`;
      }
    } catch (err) {
      partsMono.length = 0;
      statusEl.textContent = `統合エラー: ${err.message}`;
    } finally {
      mergeBtn.disabled = selectedMergeFiles.length < 2;
      runBtn.disabled = !selectedFile;
    }
  });
}

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
  await triggerDownloadOrShare({ isAuto: false, isFinalSegment: false });
});
