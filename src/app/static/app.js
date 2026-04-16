const cameraPreview = document.getElementById("camera-preview");
const captureCanvas = document.getElementById("capture-canvas");
const landmarkOverlay = document.getElementById("landmark-overlay");
const cameraStatus = document.getElementById("camera-status");
const cameraMessage = document.getElementById("camera-message");
const startCameraButton = document.getElementById("start-camera");
const stopCameraButton = document.getElementById("stop-camera");
const resetStateButton = document.getElementById("reset-state");
const deleteLastLetterButton = document.getElementById("delete-last-letter");
const uploadForm = document.getElementById("upload-form");
const wordUploadForm = document.getElementById("word-upload-form");
const startWordRecordingButton = document.getElementById("start-word-recording");
const stopWordRecordingButton = document.getElementById("stop-word-recording");
const wordRecorderStatus = document.getElementById("word-recorder-status");
const wordRecordingPreview = document.getElementById("word-recording-preview");
const speakWordButton = document.getElementById("speak-word");
const autoSpeakCheckbox = document.getElementById("auto-speak");
const speechStatus = document.getElementById("speech-status");
const speechMessage = document.getElementById("speech-message");
const speechPlayer = document.getElementById("speech-player");
const wordMessage = document.getElementById("word-message");
const wordRecordingTimer = document.getElementById("word-recording-timer");
const wordTimerElapsed = document.getElementById("word-timer-elapsed");
const wordResultCard = document.getElementById("word-result-card");
const wordResultLabel = document.getElementById("word-result-label");
const wordResultConfidenceBadge = document.getElementById("word-result-confidence-badge");
const wordResultAlt = document.getElementById("word-result-alt");
const wordPredictionHistory = document.getElementById("word-prediction-history");
const wordVocabSize = document.getElementById("word-vocab-size");
const modelInfoFooter = document.getElementById("model-info-footer");
const modelInfoText = document.getElementById("model-info-text");
const startSequenceRecordingButton = document.getElementById("start-sequence-recording");
const stopSequenceRecordingButton = document.getElementById("stop-sequence-recording");
const sequenceRecorderStatus = document.getElementById("sequence-recorder-status");
const sequenceRecordingPreview = document.getElementById("sequence-recording-preview");
const sequenceRecordingTimer = document.getElementById("sequence-recording-timer");
const sequenceTimerElapsed = document.getElementById("sequence-timer-elapsed");
const sequenceUploadForm = document.getElementById("sequence-upload-form");
const sequenceResultCard = document.getElementById("sequence-result-card");
const sequenceTranscriptEl = document.getElementById("sequence-transcript");
const sequenceResultCounts = document.getElementById("sequence-result-counts");
const sequenceChipList = document.getElementById("sequence-chip-list");
const sequenceMessage = document.getElementById("sequence-message");
const speakSequenceButton = document.getElementById("speak-sequence");
const sequenceProgress = document.getElementById("sequence-progress");
const seqExtractMode = document.getElementById("seq-extract-mode");
const seqSampleFps = document.getElementById("seq-sample-fps");
const seqDuration = document.getElementById("seq-duration");
const seqProcessingTime = document.getElementById("seq-processing-time");
const sequenceDiagnostics = document.getElementById("sequence-diagnostics");
const healthBanner = document.getElementById("health-banner");

const statTargets = {
  mode: document.getElementById("mode-value"),
  status: document.getElementById("status-value"),
  current_letter: document.getElementById("current-letter-value"),
  confidence: document.getElementById("confidence-value"),
  accepted_letters: document.getElementById("accepted-letters-value"),
  final_word: document.getElementById("final-word-value"),
  final_word_status: document.getElementById("final-word-status-value"),
  word_history: document.getElementById("word-history-value"),
  detected_hands: document.getElementById("detected-hands-value"),
  inference_fps: document.getElementById("inference-fps-value"),
  feature_dim: document.getElementById("feature-dim-value"),
};
const wordStatTargets = {
  status: document.getElementById("word-status-value"),
  predicted_word: document.getElementById("predicted-word-value"),
  confidence: document.getElementById("word-confidence-value"),
  detected_steps: document.getElementById("word-detected-steps-value"),
  sampled_steps: document.getElementById("word-sampled-steps-value"),
};

function setText(el, text) {
  if (el) {
    el.textContent = text;
  }
}

/** Idle label for the shared speak button (differs on letters vs words pages). */
let speakWordButtonIdleLabel = "Speak Phrase";

const TARGET_INFERENCE_INTERVAL_MS = 125;
const INFERENCE_MAX_WIDTH = 512;
const INFERENCE_IMAGE_TYPE = "image/jpeg";
const INFERENCE_JPEG_QUALITY = 0.72;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

let mediaStream = null;
let inferenceLoopActive = false;
let requestInFlight = false;
let lastSpokenWordKey = "";
let speechPreloadPromise = null;
let completedInferenceTimes = [];
let wordMediaRecorder = null;
let wordRecordingChunks = [];
let wordRecordingObjectUrl = null;
let wordRecordingMimeType = "";
let wordRecordingStartedAt = 0;
let wordRecordingTimerInterval = null;
let wordSessionHistory = [];
let sequenceMediaRecorder = null;
let sequenceRecordingChunks = [];
let sequenceRecordingObjectUrl = null;
let sequenceRecordingMimeType = "";
let sequenceRecordingStartedAt = 0;
let sequenceRecordingTimerInterval = null;
let lastSequenceTranscript = "";
/** When TTS audio is served as a blob URL, revoke on next playback. */
let lastSpeechObjectUrl = null;

function showSequenceProgress(step) {
  if (!sequenceProgress) return;
  sequenceProgress.hidden = false;
  const steps = sequenceProgress.querySelectorAll(".progress-step");
  let foundActive = false;
  steps.forEach((el) => {
    const s = el.dataset.step;
    if (s === step) {
      el.classList.add("is-active");
      el.classList.remove("is-done");
      foundActive = true;
    } else if (!foundActive) {
      el.classList.add("is-done");
      el.classList.remove("is-active");
    } else {
      el.classList.remove("is-active", "is-done");
    }
  });
}

function hideSequenceProgress() {
  if (!sequenceProgress) return;
  sequenceProgress.hidden = true;
  sequenceProgress.querySelectorAll(".progress-step").forEach((el) => {
    el.classList.remove("is-active", "is-done");
  });
}

function showHealthBanner(message) {
  if (!healthBanner) return;
  healthBanner.textContent = message;
  healthBanner.hidden = false;
}

function setSpeakButtonBusy(isBusy) {
  if (!speakWordButton) {
    return;
  }
  speakWordButton.disabled = isBusy;
  speakWordButton.textContent = isBusy ? "Preparing Speech..." : speakWordButtonIdleLabel;
}

function setCameraUiState({ pill, message, running }) {
  if (!cameraStatus || !cameraMessage || !startCameraButton || !stopCameraButton) {
    return;
  }
  cameraStatus.textContent = pill;
  cameraMessage.textContent = message;
  startCameraButton.disabled = running;
  stopCameraButton.disabled = !running;
}

function setWordRecorderUiState(status, message) {
  if (wordRecorderStatus) {
    wordRecorderStatus.textContent = status;
  }
  if (wordMessage && message) {
    wordMessage.textContent = message;
  }
  const isRecording = status === "recording";
  if (startWordRecordingButton) {
    startWordRecordingButton.disabled = isRecording;
  }
  if (stopWordRecordingButton) {
    stopWordRecordingButton.disabled = !isRecording;
  }
}

function recordCompletedInference() {
  const now = performance.now();
  completedInferenceTimes.push(now);
  completedInferenceTimes = completedInferenceTimes.filter((timestamp) => now - timestamp <= 2000);
  const fps = completedInferenceTimes.length / 2;
  if (statTargets.inference_fps) {
    statTargets.inference_fps.textContent = fps.toFixed(1);
  }
}

function applyPrediction(payload) {
  setText(statTargets.mode, payload.mode ?? "letters");
  setText(statTargets.status, payload.status ?? "idle");
  setText(statTargets.current_letter, payload.current_letter ?? "None");
  setText(statTargets.confidence, Number(payload.confidence ?? 0).toFixed(2));
  setText(statTargets.accepted_letters, payload.accepted_letters || "Empty");
  setText(statTargets.final_word, payload.final_word ?? "None");
  setText(statTargets.final_word_status, payload.final_word_status ?? "None");
  setText(statTargets.word_history, payload.word_history?.length ? payload.word_history.join(" | ") : "Empty");
  setText(statTargets.detected_hands, String(payload.detected_hands ?? 0));
  setText(statTargets.feature_dim, String(payload.feature_dim ?? "63"));
  const handsLandmarks = Array.isArray(payload.hands_landmarks) && payload.hands_landmarks.length
    ? payload.hands_landmarks
    : (Array.isArray(payload.hand_landmarks) && payload.hand_landmarks.length ? [payload.hand_landmarks] : []);
  drawHandLandmarks(handsLandmarks, payload.primary_hand_index ?? null);

  const word = normalizeSpeakableWord(payload.final_word);
  const status = payload.final_word_status ?? "";
  if (autoSpeakCheckbox?.checked && word && status !== "unknown") {
    const wordKey = `${word}:${status}`;
    if (wordKey !== lastSpokenWordKey) {
      speakWord(word);
      lastSpokenWordKey = wordKey;
    }
  }
}

function applyWordPrediction(payload) {
  stopRecordingTimer();
  setWordRecorderUiState("ready", "");
  setText(wordStatTargets.status, payload.status ?? "idle");
  setText(wordStatTargets.predicted_word, payload.predicted_word_display ?? payload.predicted_word ?? "None");
  setText(wordStatTargets.confidence, Number(payload.confidence ?? 0).toFixed(2));
  setText(wordStatTargets.detected_steps, String(payload.detected_steps ?? 0));
  setText(wordStatTargets.sampled_steps, String(payload.sampled_steps ?? 0));

  const level = payload.confidence_level ?? "none";
  const displayWord = payload.predicted_word_display ?? payload.predicted_word ?? null;
  const accepted = Boolean(payload.accepted_prediction);
  const topPreds = Array.isArray(payload.top_predictions) ? payload.top_predictions : [];
  const runnerUp = topPreds.length > 1 ? topPreds[1] : null;

  if (wordResultCard) {
    if ((payload.status === "word_predicted" || payload.status?.startsWith("rejected_")) && displayWord) {
      wordResultCard.hidden = false;
      wordResultLabel.textContent = accepted ? displayWord : `Suggestion: ${displayWord}`;
      wordResultConfidenceBadge.textContent = accepted
        ? `${(Number(payload.confidence) * 100).toFixed(0)}% ${level}`
        : "rejected";
      wordResultConfidenceBadge.dataset.level = accepted ? level : "low";

      if (!accepted && payload.rejection_reason === "top_predictions_too_close" && runnerUp) {
        wordResultAlt.textContent =
          `Too close to call: ${displayWord} vs ${runnerUp.label} (${(Number(payload.top_prediction_margin ?? 0) * 100).toFixed(0)}% margin).`;
      } else if (runnerUp) {
        wordResultAlt.textContent = `Runner-up: ${runnerUp.label} (${(Number(runnerUp.confidence) * 100).toFixed(0)}%)`;
      } else {
        wordResultAlt.textContent = "";
      }
    } else if (payload.status === "no_hand_detected") {
      wordResultCard.hidden = false;
      wordResultLabel.textContent = "No hand detected";
      wordResultConfidenceBadge.textContent = "—";
      wordResultConfidenceBadge.dataset.level = "none";
      wordResultAlt.textContent = "Make sure your hand is clearly visible and well-lit.";
    } else {
      wordResultCard.hidden = true;
    }
  }

  if (wordMessage) {
    if ((payload.status === "word_predicted" || payload.status?.startsWith("rejected_")) && displayWord) {
      if (!accepted && payload.rejection_reason === "top_predictions_too_close" && runnerUp) {
        wordMessage.textContent = `Uncertain result: the clip looks like either ${displayWord} or ${runnerUp.label}. Record again for a cleaner decision.`;
      } else if (!accepted) {
        wordMessage.textContent = `Rejected low-confidence result. Top suggestion was ${displayWord}, but the model was not confident enough to accept it.`;
      } else if (level === "low") {
        wordMessage.textContent = `Low confidence prediction: ${displayWord}. Try recording the sign again with a clearer hand position.`;
      } else {
        wordMessage.textContent = `Recognized as ${displayWord} (${level} confidence) from ${payload.detected_steps}/${payload.sampled_steps} sampled steps.`;
      }
    } else if (payload.status === "no_hand_detected") {
      wordMessage.textContent = "No reliable hand landmarks were detected. Check lighting and hand visibility.";
    } else {
      wordMessage.textContent = "Word clip processed.";
    }
  }

  if (payload.status === "word_predicted" && accepted && displayWord) {
    wordSessionHistory.push({
      word: displayWord,
      confidence: Number(payload.confidence ?? 0),
      level,
    });
    renderWordHistory();
  }
}

function renderWordHistory() {
  if (!wordPredictionHistory) {
    return;
  }
  wordPredictionHistory.innerHTML = "";
  for (const entry of wordSessionHistory) {
    const li = document.createElement("li");
    li.textContent = `${entry.word} — ${(entry.confidence * 100).toFixed(0)}%`;
    if (entry.level === "low") {
      li.classList.add("hist-low");
    }
    wordPredictionHistory.appendChild(li);
  }
}

function setSpeechUiState(status, message) {
  if (speechStatus) {
    speechStatus.textContent = status;
  }
  if (speechMessage) {
    speechMessage.textContent = message;
  }
}

function normalizeSpeakableWord(value) {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed === "None") {
    return "";
  }
  return trimmed;
}

function getSpeakText() {
  const finalWord = normalizeSpeakableWord(statTargets.final_word?.textContent ?? "");
  if (finalWord) {
    return finalWord;
  }
  return normalizeSpeakableWord(statTargets.accepted_letters?.textContent ?? "");
}

function getPhraseText() {
  const historyText = normalizeSpeakableWord(statTargets.word_history?.textContent ?? "").replaceAll("|", " ");
  const currentBuffer = normalizeSpeakableWord(statTargets.accepted_letters?.textContent ?? "");
  return [historyText, currentBuffer].filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

function stopSpeechPlayback() {
  if (lastSpeechObjectUrl) {
    URL.revokeObjectURL(lastSpeechObjectUrl);
    lastSpeechObjectUrl = null;
  }
  if (speechPlayer) {
    speechPlayer.pause();
    speechPlayer.removeAttribute("src");
    speechPlayer.load();
  }
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

async function preloadSpeechEngine() {
  if (speechPreloadPromise) {
    return speechPreloadPromise;
  }

  setSpeechUiState("warming", "Preparing the Piper voice. The first TTS request can take a couple of seconds.");
  speechPreloadPromise = fetch("/api/tts/preload", {
    method: "POST",
  })
    .then(async (response) => {
      if (!response.ok) {
        let detail = `Speech preload failed with ${response.status}`;
        try {
          const payload = await response.json();
          if (payload?.detail) {
            detail = String(payload.detail);
          }
        } catch (_error) {
          // Ignore JSON parsing failures and keep the generic error.
        }
        throw new Error(detail);
      }
      setSpeechUiState("ready", "Piper voice is ready.");
      return true;
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : "Speech preload failed";
      setSpeechUiState("error", message);
      throw error;
    });

  return speechPreloadPromise;
}

function buildUtterance(word) {
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.lang = "en-US";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  return utterance;
}

async function playServerSpeech(word = null) {
  const normalized = typeof word === "string" ? normalizeSpeakableWord(word) : "";
  let phraseText = word === null ? getPhraseText() : normalized;
  if (phraseText) {
    phraseText = phraseText.replace(/\s+/g, " ").trim();
  }
  if (!phraseText) {
    setSpeechUiState("idle", "No phrase is available for speech yet.");
    return false;
  }
  if (!speechPlayer) {
    throw new Error("Speech player element is not available");
  }

  /** Explicit client text (e.g. multi-word sequence transcript) must be sent in the POST body.
   * Relying on GET query params for phrases is fragile for multi-word strings and falls back to
   * the letters pipeline when ``text`` is missing, which is empty on the sequence page. */
  const textPayload = word === null ? null : phraseText;

  setSpeechUiState("loading", `Generating audio for: ${phraseText}`);
  await preloadSpeechEngine();
  stopSpeechPlayback();

  const response = await fetch("/api/tts/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textPayload }),
  });

  if (!response.ok) {
    let detail = `TTS failed with ${response.status}`;
    try {
      const err = await response.json();
      if (err.detail) {
        detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      }
    } catch (_error) {
      // Ignore JSON parse errors.
    }
    throw new Error(detail);
  }

  const blob = await response.blob();
  lastSpeechObjectUrl = URL.createObjectURL(blob);
  speechPlayer.src = lastSpeechObjectUrl;
  speechPlayer.autoplay = true;
  speechPlayer.load();

  try {
    const playPromise = speechPlayer.play();
    if (playPromise && typeof playPromise.then === "function") {
      await playPromise;
    }
    setSpeechUiState("speaking", `Speaking: ${phraseText}`);
    return true;
  } catch (error) {
    setSpeechUiState("ready", `Audio is ready for ${phraseText}. Press play on the built-in player if autoplay was blocked.`);
    return true;
  }
}

function speakWordInBrowser(word) {
  const normalized = normalizeSpeakableWord(word);
  if (!normalized) {
    setSpeechUiState("idle", "No finalized word is available for speech yet.");
    return false;
  }

  if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
    setSpeechUiState("unsupported", "This browser does not support speech synthesis.");
    return false;
  }

  stopSpeechPlayback();
  const utterance = buildUtterance(normalized);
  utterance.onstart = () => {
    setSpeechUiState("speaking", `Speaking: ${normalized}`);
  };
  utterance.onend = () => {
    setSpeechUiState("idle", `Finished speaking: ${normalized}`);
  };
  utterance.onerror = () => {
    setSpeechUiState("error", `Speech synthesis failed for: ${normalized}`);
  };
  window.speechSynthesis.speak(utterance);
  return true;
}

async function speakWord(word) {
  const normalized = normalizeSpeakableWord(word);
  if (!normalized) {
    setSpeechUiState("idle", "No finalized word or accepted letters are available for speech yet.");
    return false;
  }

  try {
    await playServerSpeech(normalized);
    return true;
  } catch (error) {
    const serverMessage = error instanceof Error ? error.message : "Server TTS failed";
    if (("speechSynthesis" in window) && typeof SpeechSynthesisUtterance !== "undefined") {
      setSpeechUiState("fallback", `${serverMessage}. Falling back to browser speech.`);
      return speakWordInBrowser(normalized);
    }

    setSpeechUiState("error", serverMessage);
    return false;
  }
}

async function speakPhrase() {
  const phraseText = getPhraseText();
  if (!phraseText) {
    setSpeechUiState("idle", "No phrase is available for speech yet.");
    return false;
  }

  try {
    await playServerSpeech(null);
    return true;
  } catch (error) {
    const serverMessage = error instanceof Error ? error.message : "Server TTS failed";
    if (("speechSynthesis" in window) && typeof SpeechSynthesisUtterance !== "undefined") {
      setSpeechUiState("fallback", `${serverMessage}. Falling back to browser speech.`);
      return speakWordInBrowser(phraseText);
    }

    setSpeechUiState("error", serverMessage);
    return false;
  }
}

function drawHandLandmarks(handsLandmarks, primaryHandIndex = null) {
  if (!landmarkOverlay) {
    return;
  }

  const width = cameraPreview.videoWidth || 640;
  const height = cameraPreview.videoHeight || 480;
  landmarkOverlay.width = width;
  landmarkOverlay.height = height;

  const context = landmarkOverlay.getContext("2d");
  context.clearRect(0, 0, width, height);

  if (!Array.isArray(handsLandmarks) || handsLandmarks.length === 0) {
    return;
  }

  const palette = [
    { stroke: "rgba(90, 222, 173, 0.95)", fill: "rgba(255, 244, 92, 0.95)" },
    { stroke: "rgba(70, 147, 255, 0.95)", fill: "rgba(255, 151, 112, 0.95)" },
  ];

  const radius = Math.max(3, width / 140);
  handsLandmarks.forEach((landmarks, handIndex) => {
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return;
    }

    const isPrimary = primaryHandIndex === handIndex;
    const colors = palette[handIndex % palette.length];
    context.lineWidth = isPrimary ? Math.max(3, width / 220) : Math.max(2, width / 260);
    context.strokeStyle = colors.stroke;
    context.fillStyle = colors.fill;

    for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
      const start = landmarks[startIndex];
      const end = landmarks[endIndex];
      if (!start || !end) {
        continue;
      }

      context.beginPath();
      context.moveTo(start[0] * width, start[1] * height);
      context.lineTo(end[0] * width, end[1] * height);
      context.stroke();
    }

    for (const landmark of landmarks) {
      if (!Array.isArray(landmark) || landmark.length < 2) {
        continue;
      }
      context.beginPath();
      context.arc(landmark[0] * width, landmark[1] * height, radius, 0, Math.PI * 2);
      context.fill();
    }
  });
}

async function postFrame(blob, filename = "camera-frame.png") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  const response = await fetch("/api/letters/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Inference request failed with ${response.status}`);
  }

  return response.json();
}

async function postWordVideo(blob, filename = "word-clip.mp4") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  const response = await fetch("/api/words/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Word inference request failed with ${response.status}`);
  }

  return response.json();
}

function resetWordRecordingPreview() {
  if (wordRecordingObjectUrl) {
    URL.revokeObjectURL(wordRecordingObjectUrl);
    wordRecordingObjectUrl = null;
  }
  if (wordRecordingPreview) {
    wordRecordingPreview.pause();
    wordRecordingPreview.removeAttribute("src");
    wordRecordingPreview.load();
  }
}

function selectWordRecordingMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];
  for (const mimeType of candidates) {
    if (!MediaRecorder.isTypeSupported || MediaRecorder.isTypeSupported(mimeType)) {
      return mimeType;
    }
  }
  return "";
}

function buildWordRecordingFilename() {
  const extension = wordRecordingMimeType.includes("mp4") ? "mp4" : "webm";
  return `word-clip-${Date.now()}.${extension}`;
}

async function ensureCameraStream() {
  if (mediaStream) {
    return;
  }
  await startCameraStream();
  if (!mediaStream) {
    throw new Error("Camera stream is unavailable");
  }
}

function stopWordRecordingRecorder() {
  if (wordMediaRecorder && wordMediaRecorder.state !== "inactive") {
    wordMediaRecorder.stop();
  }
}

function startRecordingTimer() {
  stopRecordingTimer();
  if (wordRecordingTimer) {
    wordRecordingTimer.hidden = false;
  }
  wordRecordingTimerInterval = setInterval(() => {
    if (wordTimerElapsed) {
      const elapsed = (performance.now() - wordRecordingStartedAt) / 1000;
      wordTimerElapsed.textContent = `${elapsed.toFixed(1)}s`;
    }
  }, 100);
}

function stopRecordingTimer() {
  if (wordRecordingTimerInterval !== null) {
    clearInterval(wordRecordingTimerInterval);
    wordRecordingTimerInterval = null;
  }
  if (wordRecordingTimer) {
    wordRecordingTimer.hidden = true;
  }
}

async function startWordRecording() {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("This browser does not support in-page video recording.");
  }
  if (wordMediaRecorder && wordMediaRecorder.state === "recording") {
    return;
  }

  await ensureCameraStream();
  wordRecordingMimeType = selectWordRecordingMimeType();
  if (!wordRecordingMimeType) {
    throw new Error("No supported video recording format is available in this browser.");
  }

  resetWordRecordingPreview();
  wordRecordingChunks = [];
  wordMediaRecorder = new MediaRecorder(mediaStream, { mimeType: wordRecordingMimeType });
  wordMediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      wordRecordingChunks.push(event.data);
    }
  });
  wordMediaRecorder.addEventListener("stop", async () => {
    const durationSeconds = Math.max(0, (performance.now() - wordRecordingStartedAt) / 1000);
    const blob = new Blob(wordRecordingChunks, { type: wordRecordingMimeType || "video/webm" });
    if (!blob.size) {
      setWordRecorderUiState("error", "The recorded clip was empty. Try recording the sign again.");
      return;
    }

    wordRecordingObjectUrl = URL.createObjectURL(blob);
    if (wordRecordingPreview) {
      wordRecordingPreview.src = wordRecordingObjectUrl;
      wordRecordingPreview.load();
    }

    setWordRecorderUiState("analyzing", `Recorded ${durationSeconds.toFixed(1)}s clip. Extracting temporal landmarks...`);
    try {
      const payload = await postWordVideo(blob, buildWordRecordingFilename());
      applyWordPrediction(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Recorded clip inference failed";
      setWordRecorderUiState("error", message);
      setText(wordStatTargets.status, "error");
    }
  }, { once: true });

  wordRecordingStartedAt = performance.now();
  wordMediaRecorder.start();
  startRecordingTimer();
  setWordRecorderUiState("recording", "Recording — perform the sign now. Aim for 1–3 seconds, then press Stop & Analyze.");
}

async function runInferenceTick() {
  if (!cameraPreview || !captureCanvas || !mediaStream || requestInFlight
    || cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  requestInFlight = true;
  try {
    const { width, height } = getInferenceFrameSize();
    captureCanvas.width = width;
    captureCanvas.height = height;

    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraPreview, 0, 0, width, height);

    const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, INFERENCE_IMAGE_TYPE, INFERENCE_JPEG_QUALITY));
    if (!blob) {
      throw new Error("Failed to capture a frame from the camera");
    }

    const payload = await postFrame(blob, "camera-frame.jpg");
    applyPrediction(payload);
    recordCompletedInference();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown camera inference error";
    setCameraUiState({
      pill: "error",
      message,
      running: Boolean(mediaStream),
    });
  } finally {
    requestInFlight = false;
  }
}

function stopCameraStream() {
  inferenceLoopActive = false;
  stopRecordingTimer();
  stopSequenceTimer();
  stopWordRecordingRecorder();
  stopSequenceRecordingRecorder();

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (cameraPreview) {
    cameraPreview.srcObject = null;
  }
  completedInferenceTimes = [];
  if (statTargets.inference_fps) {
    statTargets.inference_fps.textContent = "0.0";
  }
  drawHandLandmarks([]);
  const page = document.body?.dataset?.page;
  const idleMessage = page === "letters"
    ? "Camera is off. Start the stream to run live inference from your browser."
    : "Camera is off. Start the stream when you are ready to record.";
  setCameraUiState({
    pill: "idle",
    message: idleMessage,
    running: false,
  });
}

function getInferenceFrameSize() {
  const sourceWidth = cameraPreview?.videoWidth || 640;
  const sourceHeight = cameraPreview?.videoHeight || 480;
  const scale = Math.min(1, INFERENCE_MAX_WIDTH / sourceWidth);
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

async function startCameraStream() {
  if (!cameraPreview || !captureCanvas) {
    return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960, max: 1280 },
        height: { ideal: 720, max: 960 },
        facingMode: "user",
      },
      audio: false,
    });

    cameraPreview.srcObject = mediaStream;
    await cameraPreview.play();

    const isLettersPage = document.body?.dataset?.page === "letters";
    setCameraUiState({
      pill: "running",
      message: isLettersPage
        ? "Camera is live. Letter frames are sampled at about 8 FPS."
        : "Camera is live. Press a Start Recording button when you're ready.",
      running: true,
    });

    // Letter-frame inference is the only mode that continuously posts
    // frames to the server. Word/sequence pages only need the stream for
    // MediaRecorder, so skip the loop outside the letters page.
    if (isLettersPage) {
      void startInferenceLoop();
    }
  } catch (error) {
    stopCameraStream();
    const message = error instanceof Error ? error.message : "Unable to access the camera";
    setCameraUiState({
      pill: "error",
      message,
      running: false,
    });
  }
}

async function startInferenceLoop() {
  if (inferenceLoopActive) {
    return;
  }

  inferenceLoopActive = true;
  while (mediaStream && inferenceLoopActive) {
    const startedAt = performance.now();
    await runInferenceTick();
    const elapsedMs = performance.now() - startedAt;
    const delayMs = Math.max(0, TARGET_INFERENCE_INTERVAL_MS - elapsedMs);
    if (delayMs > 0) {
      await new Promise((resolve) => window.setTimeout(resolve, delayMs));
    }
  }
}

async function resetPipelineState() {
  const response = await fetch("/api/reset", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Reset failed with ${response.status}`);
  }

  const payload = await response.json();
  applyPrediction({
    mode: "letters",
    status: payload.status,
    current_letter: null,
    confidence: 0,
    accepted_letters: payload.accepted_letters,
    final_word: null,
    final_word_status: null,
    word_history: payload.word_history,
    detected_hands: 0,
    primary_hand_index: null,
    feature_dim: statTargets.feature_dim?.textContent ?? "63",
    hand_landmarks: [],
    hands_landmarks: [],
  });
  lastSpokenWordKey = "";
  stopSpeechPlayback();
  setSpeechUiState("idle", "Buffer reset. Waiting for a new finalized word.");
}

async function deleteLastAcceptedLetter() {
  const response = await fetch("/api/delete-last", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Delete-last failed with ${response.status}`);
  }

  const payload = await response.json();
  applyPrediction({
    mode: "letters",
    status: payload.status,
    current_letter: null,
    confidence: 0,
    accepted_letters: payload.accepted_letters,
    final_word: statTargets.final_word?.textContent ?? null,
    final_word_status: statTargets.final_word_status?.textContent ?? null,
    word_history: payload.word_history,
    detected_hands: Number(statTargets.detected_hands?.textContent || 0),
    primary_hand_index: null,
    feature_dim: statTargets.feature_dim?.textContent ?? "63",
    hand_landmarks: [],
    hands_landmarks: [],
  });
}

startCameraButton?.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraUiState({
      pill: "unsupported",
      message: "This browser does not support getUserMedia camera capture.",
      running: false,
    });
    return;
  }

  if (mediaStream) {
    return;
  }

  await startCameraStream();
});

stopCameraButton?.addEventListener("click", () => {
  stopCameraStream();
});

resetStateButton?.addEventListener("click", async () => {
  try {
    await resetPipelineState();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Reset failed";
    setCameraUiState({
      pill: "error",
      message,
      running: Boolean(mediaStream),
    });
  }
});

deleteLastLetterButton?.addEventListener("click", async () => {
  try {
    await deleteLastAcceptedLetter();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Delete-last failed";
    setCameraUiState({
      pill: "error",
      message,
      running: Boolean(mediaStream),
    });
  }
});

uploadForm?.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = uploadForm.querySelector("input[type='file']");
  const file = fileInput?.files?.[0];
  if (!file) {
    return;
  }

  try {
    const payload = await postFrame(file, file.name);
    applyPrediction(payload);
    if (cameraMessage) {
      cameraMessage.textContent = "Manual frame uploaded successfully.";
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Manual upload failed";
    setCameraUiState({
      pill: "error",
      message,
      running: Boolean(mediaStream),
    });
  }
});

startWordRecordingButton?.addEventListener("click", async () => {
  try {
    await startWordRecording();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to start word recording";
    setWordRecorderUiState("error", message);
  }
});

stopWordRecordingButton?.addEventListener("click", () => {
  stopWordRecordingRecorder();
});

wordUploadForm?.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = wordUploadForm.querySelector("input[type='file']");
  const file = fileInput?.files?.[0];
  if (!file) {
    return;
  }

  const submitButton = wordUploadForm.querySelector("button[type='submit']");
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
  }
  if (wordMessage) {
    wordMessage.textContent = "Uploading the word clip and extracting temporal landmarks...";
  }
  setWordRecorderUiState("uploading", "");

  try {
    const payload = await postWordVideo(file, file.name);
    applyWordPrediction(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Word clip upload failed";
    if (wordMessage) {
      wordMessage.textContent = message;
    }
    setWordRecorderUiState("error", "");
    setText(wordStatTargets.status, "error");
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = "Run Word Clip";
    }
  }
});

speechPlayer?.addEventListener("ended", () => {
  const currentWord = getSpeakText();
  setSpeechUiState("idle", currentWord ? `Finished speaking: ${currentWord}` : "Finished speaking.");
});

speechPlayer?.addEventListener("canplay", () => {
  const currentWord = getSpeakText();
  setSpeechUiState("ready", currentWord ? `Audio is ready for: ${currentWord}` : "Audio is ready.");
});

speechPlayer?.addEventListener("error", () => {
  setSpeechUiState("error", "Audio playback failed. Try clicking play on the built-in player or regenerate the word.");
});

function resolvePrimarySpeakTarget() {
  const page = document.body?.dataset?.page;
  if (page === "words") {
    const label = wordResultLabel?.textContent?.replace(/^Suggestion:\s*/i, "").trim();
    if (label && label !== "—") {
      return label;
    }
    if (wordSessionHistory.length > 0) {
      return wordSessionHistory[wordSessionHistory.length - 1].word;
    }
    return "";
  }
  return getPhraseText();
}

speakWordButton?.addEventListener("click", async () => {
  const page = document.body?.dataset?.page;
  const target = resolvePrimarySpeakTarget();
  setSpeechUiState(
    "clicked",
    target ? `Speech request started for: ${target}` : "Speech button clicked, but there is no phrase yet.",
  );
  setSpeakButtonBusy(true);
  try {
    let spoke = false;
    if (page === "words" && target) {
      spoke = await speakWord(target);
    } else {
      spoke = await speakPhrase();
    }
    if (spoke) {
      const currentWord = resolvePrimarySpeakTarget();
      const currentStatus = statTargets.final_word_status?.textContent ?? "";
      lastSpokenWordKey = currentWord ? `${currentWord}:${currentStatus}` : "";
    }
  } finally {
    setSpeakButtonBusy(false);
  }
});

window.addEventListener("beforeunload", () => {
  stopSpeechPlayback();
  resetWordRecordingPreview();
  resetSequenceRecordingPreview();
  stopCameraStream();
});

function setSequenceRecorderUiState(status, message) {
  if (sequenceRecorderStatus) {
    sequenceRecorderStatus.textContent = status;
  }
  if (sequenceMessage && typeof message === "string" && message.length > 0) {
    sequenceMessage.textContent = message;
  }
  const isRecording = status === "recording";
  if (startSequenceRecordingButton) {
    startSequenceRecordingButton.disabled = isRecording;
  }
  if (stopSequenceRecordingButton) {
    stopSequenceRecordingButton.disabled = !isRecording;
  }
}

function startSequenceTimer() {
  stopSequenceTimer();
  if (sequenceRecordingTimer) {
    sequenceRecordingTimer.hidden = false;
  }
  sequenceRecordingTimerInterval = setInterval(() => {
    if (sequenceTimerElapsed) {
      const elapsed = (performance.now() - sequenceRecordingStartedAt) / 1000;
      sequenceTimerElapsed.textContent = `${elapsed.toFixed(1)}s`;
    }
  }, 100);
}

function stopSequenceTimer() {
  if (sequenceRecordingTimerInterval !== null) {
    clearInterval(sequenceRecordingTimerInterval);
    sequenceRecordingTimerInterval = null;
  }
  if (sequenceRecordingTimer) {
    sequenceRecordingTimer.hidden = true;
  }
}

function resetSequenceRecordingPreview() {
  if (sequenceRecordingObjectUrl) {
    URL.revokeObjectURL(sequenceRecordingObjectUrl);
    sequenceRecordingObjectUrl = null;
  }
  if (sequenceRecordingPreview) {
    sequenceRecordingPreview.pause();
    sequenceRecordingPreview.removeAttribute("src");
    sequenceRecordingPreview.load();
  }
}

function buildSequenceRecordingFilename() {
  const extension = sequenceRecordingMimeType.includes("mp4") ? "mp4" : "webm";
  return `word-sequence-${Date.now()}.${extension}`;
}

let lastSequenceStartTime = 0;

async function postSequenceVideo(blob, filename) {
  showSequenceProgress("upload");
  lastSequenceStartTime = performance.now();
  const formData = new FormData();
  formData.append("file", blob, filename);

  showSequenceProgress("landmarks");
  const response = await fetch("/api/words/sequence", {
    method: "POST",
    body: formData,
  });

  showSequenceProgress("classify");

  if (!response.ok) {
    hideSequenceProgress();
    throw new Error(`Sequence inference request failed with ${response.status}`);
  }

  const payload = await response.json();
  hideSequenceProgress();
  return payload;
}

function stopSequenceRecordingRecorder() {
  if (sequenceMediaRecorder && sequenceMediaRecorder.state !== "inactive") {
    sequenceMediaRecorder.stop();
  }
}

async function startSequenceRecording() {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("This browser does not support in-page video recording.");
  }
  if (sequenceMediaRecorder && sequenceMediaRecorder.state === "recording") {
    return;
  }

  await ensureCameraStream();
  sequenceRecordingMimeType = selectWordRecordingMimeType();
  if (!sequenceRecordingMimeType) {
    throw new Error("No supported video recording format is available in this browser.");
  }

  resetSequenceRecordingPreview();
  sequenceRecordingChunks = [];
  sequenceMediaRecorder = new MediaRecorder(mediaStream, { mimeType: sequenceRecordingMimeType });
  sequenceMediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      sequenceRecordingChunks.push(event.data);
    }
  });
  sequenceMediaRecorder.addEventListener("stop", async () => {
    const durationSeconds = Math.max(0, (performance.now() - sequenceRecordingStartedAt) / 1000);
    stopSequenceTimer();
    const blob = new Blob(sequenceRecordingChunks, { type: sequenceRecordingMimeType || "video/webm" });
    if (!blob.size) {
      setSequenceRecorderUiState("error", "The recorded clip was empty. Try recording again.");
      return;
    }

    sequenceRecordingObjectUrl = URL.createObjectURL(blob);
    if (sequenceRecordingPreview) {
      sequenceRecordingPreview.src = sequenceRecordingObjectUrl;
      sequenceRecordingPreview.load();
    }

    setSequenceRecorderUiState(
      "analyzing",
      `Recorded ${durationSeconds.toFixed(1)}s clip. Splitting into signing segments...`,
    );
    try {
      const payload = await postSequenceVideo(blob, buildSequenceRecordingFilename());
      applySequenceResult(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Sequence clip inference failed";
      setSequenceRecorderUiState("error", message);
    }
  }, { once: true });

  sequenceRecordingStartedAt = performance.now();
  sequenceMediaRecorder.start();
  startSequenceTimer();
  setSequenceRecorderUiState(
    "recording",
    "Recording — sign multiple words in a row, pausing briefly between each. Press Stop & Analyze when done.",
  );
}

function formatTimeRange(start, end) {
  const s = Number.isFinite(start) ? Number(start) : 0;
  const e = Number.isFinite(end) ? Number(end) : 0;
  return `${s.toFixed(1)}–${e.toFixed(1)}s`;
}

function renderSequenceChips(segments) {
  if (!sequenceChipList) {
    return;
  }
  sequenceChipList.innerHTML = "";
  if (!Array.isArray(segments) || segments.length === 0) {
    return;
  }
  segments.forEach((segment, index) => {
    const li = document.createElement("li");
    li.className = "sequence-chip";
    const accepted = Boolean(segment.accepted_prediction);
    li.dataset.accepted = String(accepted);
    const level = segment.confidence_level ?? "none";
    li.dataset.level = level;

    const orderEl = document.createElement("span");
    orderEl.className = "sequence-chip-order";
    orderEl.textContent = `#${index + 1}`;

    const wordEl = document.createElement("span");
    wordEl.className = "sequence-chip-word";
    const displayWord = segment.predicted_word_display ?? segment.predicted_word ?? "—";
    wordEl.textContent = accepted ? displayWord : `(${displayWord})`;

    const confidenceEl = document.createElement("span");
    confidenceEl.className = "confidence-badge";
    confidenceEl.dataset.level = accepted ? level : "low";
    const confidencePct = Number(segment.confidence ?? 0) * 100;
    confidenceEl.textContent = accepted ? `${confidencePct.toFixed(0)}%` : "rejected";

    const timeEl = document.createElement("span");
    timeEl.className = "sequence-chip-time";
    timeEl.textContent = formatTimeRange(segment.start_time, segment.end_time);

    li.append(orderEl, wordEl, confidenceEl, timeEl);

    if (!accepted && segment.rejection_reason) {
      const reasonEl = document.createElement("span");
      reasonEl.className = "sequence-chip-reason";
      reasonEl.textContent = segment.rejection_reason === "top_predictions_too_close"
        ? "too close to call"
        : segment.rejection_reason.replaceAll("_", " ");
      li.append(reasonEl);
    }

    sequenceChipList.append(li);
  });
}

function applySequenceResult(payload) {
  stopSequenceTimer();
  hideSequenceProgress();
  setSequenceRecorderUiState("ready", "");

  const transcript = typeof payload.transcript === "string" ? payload.transcript.trim() : "";
  const segments = Array.isArray(payload.segments) ? payload.segments : [];
  lastSequenceTranscript = transcript;

  if (sequenceResultCard) {
    sequenceResultCard.hidden = false;
  }
  if (sequenceTranscriptEl) {
    sequenceTranscriptEl.textContent = transcript ? transcript : "—";
  }
  if (sequenceResultCounts) {
    const duration = Number(payload.duration_seconds ?? 0).toFixed(1);
    sequenceResultCounts.textContent =
      `${payload.accepted_segments ?? 0}/${payload.total_segments ?? 0} signs · ${duration}s clip`;
  }
  if (speakSequenceButton) {
    speakSequenceButton.disabled = !transcript;
  }
  renderSequenceChips(segments);

  const processingMs = lastSequenceStartTime ? performance.now() - lastSequenceStartTime : 0;
  if (sequenceDiagnostics) {
    sequenceDiagnostics.hidden = false;
    setText(seqExtractMode, payload.frame_extract ?? "—");
    setText(seqSampleFps, payload.sample_fps != null ? `${Number(payload.sample_fps).toFixed(1)}` : "—");
    setText(seqDuration, payload.duration_seconds != null ? `${Number(payload.duration_seconds).toFixed(1)}s` : "—");
    setText(seqProcessingTime, processingMs > 0 ? `${(processingMs / 1000).toFixed(1)}s` : "—");
  }

  if (sequenceMessage) {
    if (payload.status === "sequence_predicted" && transcript) {
      sequenceMessage.textContent =
        `Transcript assembled from ${payload.accepted_segments} accepted segment(s). Rejected segments are shown greyed out.`;
    } else if (payload.status === "sequence_inconclusive") {
      sequenceMessage.textContent =
        `Found ${segments.length} signing segment(s) but none passed the confidence threshold. Try signing more slowly with clearer pauses.`;
    } else if (payload.status === "no_hand_detected") {
      sequenceMessage.textContent =
        "No reliable hand landmarks were detected in the clip. Check lighting and hand visibility.";
    } else {
      sequenceMessage.textContent = "Sequence clip processed.";
    }
  }
}

startSequenceRecordingButton?.addEventListener("click", async () => {
  try {
    await startSequenceRecording();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to start sequence recording";
    setSequenceRecorderUiState("error", message);
  }
});

stopSequenceRecordingButton?.addEventListener("click", () => {
  stopSequenceRecordingRecorder();
});

sequenceUploadForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = sequenceUploadForm.querySelector("input[type='file']");
  const file = fileInput?.files?.[0];
  if (!file) {
    return;
  }

  const submitButton = sequenceUploadForm.querySelector("button[type='submit']");
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
  }
  setSequenceRecorderUiState("uploading", "Uploading the sequence clip and running segmentation...");

  try {
    const payload = await postSequenceVideo(file, file.name);
    applySequenceResult(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Sequence clip upload failed";
    setSequenceRecorderUiState("error", message);
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = "Run Sequence Clip";
    }
  }
});

speakSequenceButton?.addEventListener("click", async () => {
  if (!lastSequenceTranscript) {
    return;
  }
  const originalLabel = speakSequenceButton.textContent;
  speakSequenceButton.disabled = true;
  speakSequenceButton.textContent = "Preparing Speech...";
  try {
    await playServerSpeech(lastSequenceTranscript);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Server TTS failed";
    setSpeechUiState("fallback", `${message}. Falling back to browser speech.`);
    speakWordInBrowser(lastSequenceTranscript);
  } finally {
    speakSequenceButton.disabled = false;
    speakSequenceButton.textContent = originalLabel;
  }
});

async function fetchHealthInfo() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      showHealthBanner("Model info unavailable — the server returned an error. Recognition may still work.");
      return;
    }
    const data = await response.json();
    if (wordVocabSize && data.word_vocab_size) {
      wordVocabSize.textContent = String(data.word_vocab_size);
    }
    const homeVocabSize = document.getElementById("home-vocab-size");
    if (homeVocabSize && data.word_vocab_size) {
      homeVocabSize.textContent = String(data.word_vocab_size);
    }
    if (modelInfoFooter && modelInfoText) {
      const parts = [`Letters feature dim: ${data.feature_dim}`];
      if (data.word_feature_dim) {
        parts.push(`Word features: ${data.word_feature_dim}`);
      }
      if (data.word_vocab_size) {
        parts.push(`Word vocab: ${data.word_vocab_size} signs`);
      }
      modelInfoText.textContent = parts.join(" · ");
      modelInfoFooter.hidden = false;
    }
  } catch (_error) {
    showHealthBanner("Model info unavailable — could not reach the server. Recognition may still work.");
  }
}

window.addEventListener("load", () => {
  if (speakWordButton?.textContent?.trim()) {
    speakWordButtonIdleLabel = speakWordButton.textContent.trim();
  }
  setWordRecorderUiState("idle", "Record a short sign clip (1–3s) from the live camera or upload one manually.");
  setSequenceRecorderUiState(
    "idle",
    "Record or upload a longer clip of several signs. Pause briefly (~0.3s) between each sign.",
  );
  if (speakSequenceButton) {
    speakSequenceButton.disabled = true;
  }
  fetchHealthInfo();
  preloadSpeechEngine().catch(() => {});
});
