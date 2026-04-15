const cameraPreview = document.getElementById("camera-preview");
const captureCanvas = document.getElementById("capture-canvas");
const landmarkOverlay = document.getElementById("landmark-overlay");
const cameraStatus = document.getElementById("camera-status");
const cameraMessage = document.getElementById("camera-message");
const startCameraButton = document.getElementById("start-camera");
const stopCameraButton = document.getElementById("stop-camera");
const resetStateButton = document.getElementById("reset-state");
const uploadForm = document.getElementById("upload-form");
const speakWordButton = document.getElementById("speak-word");
const autoSpeakCheckbox = document.getElementById("auto-speak");
const speechStatus = document.getElementById("speech-status");
const speechMessage = document.getElementById("speech-message");
const speechPlayer = document.getElementById("speech-player");

const statTargets = {
  mode: document.getElementById("mode-value"),
  status: document.getElementById("status-value"),
  current_letter: document.getElementById("current-letter-value"),
  confidence: document.getElementById("confidence-value"),
  accepted_letters: document.getElementById("accepted-letters-value"),
  final_word: document.getElementById("final-word-value"),
  final_word_status: document.getElementById("final-word-status-value"),
  word_history: document.getElementById("word-history-value"),
  feature_dim: document.getElementById("feature-dim-value"),
};

const INFERENCE_INTERVAL_MS = 700;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

let mediaStream = null;
let inferenceTimer = null;
let requestInFlight = false;
let lastSpokenWordKey = "";
let speechPreloadPromise = null;

function setSpeakButtonBusy(isBusy) {
  if (!speakWordButton) {
    return;
  }
  speakWordButton.disabled = isBusy;
  speakWordButton.textContent = isBusy ? "Preparing Speech..." : "Speak Word or Buffer";
}

function setCameraUiState({ pill, message, running }) {
  cameraStatus.textContent = pill;
  cameraMessage.textContent = message;
  startCameraButton.disabled = running;
  stopCameraButton.disabled = !running;
}

function applyPrediction(payload) {
  statTargets.mode.textContent = payload.mode ?? "letters";
  statTargets.status.textContent = payload.status ?? "idle";
  statTargets.current_letter.textContent = payload.current_letter ?? "None";
  statTargets.confidence.textContent = Number(payload.confidence ?? 0).toFixed(2);
  statTargets.accepted_letters.textContent = payload.accepted_letters || "Empty";
  statTargets.final_word.textContent = payload.final_word ?? "None";
  statTargets.final_word_status.textContent = payload.final_word_status ?? "None";
  statTargets.word_history.textContent = payload.word_history?.length ? payload.word_history.join(" | ") : "Empty";
  statTargets.feature_dim.textContent = String(payload.feature_dim ?? "63");
  drawHandLandmarks(payload.hand_landmarks ?? []);

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

function stopSpeechPlayback() {
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

function buildServerSpeechUrl(word) {
  const params = new URLSearchParams({
    text: word,
    ts: String(Date.now()),
  });
  return `/api/tts/speak?${params.toString()}`;
}

async function playServerSpeech(word) {
  const normalized = normalizeSpeakableWord(word);
  if (!normalized) {
    setSpeechUiState("idle", "No finalized word is available for speech yet.");
    return false;
  }
  if (!speechPlayer) {
    throw new Error("Speech player element is not available");
  }

  setSpeechUiState("loading", `Generating audio for: ${normalized}`);
  await preloadSpeechEngine();
  stopSpeechPlayback();
  speechPlayer.src = buildServerSpeechUrl(normalized);
  speechPlayer.autoplay = true;
  speechPlayer.load();

  try {
    const playPromise = speechPlayer.play();
    if (playPromise && typeof playPromise.then === "function") {
      await playPromise;
    }
    setSpeechUiState("speaking", `Speaking: ${normalized}`);
    return true;
  } catch (error) {
    setSpeechUiState("ready", `Audio is ready for ${normalized}. Press play on the built-in player if autoplay was blocked.`);
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

function drawHandLandmarks(landmarks) {
  if (!landmarkOverlay) {
    return;
  }

  const width = cameraPreview.videoWidth || 640;
  const height = cameraPreview.videoHeight || 480;
  landmarkOverlay.width = width;
  landmarkOverlay.height = height;

  const context = landmarkOverlay.getContext("2d");
  context.clearRect(0, 0, width, height);

  if (!Array.isArray(landmarks) || landmarks.length === 0) {
    return;
  }

  context.lineWidth = Math.max(2, width / 240);
  context.strokeStyle = "rgba(90, 222, 173, 0.95)";
  context.fillStyle = "rgba(255, 244, 92, 0.95)";

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

  const radius = Math.max(3, width / 140);
  for (const landmark of landmarks) {
    if (!Array.isArray(landmark) || landmark.length < 2) {
      continue;
    }
    context.beginPath();
    context.arc(landmark[0] * width, landmark[1] * height, radius, 0, Math.PI * 2);
    context.fill();
  }
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

async function runInferenceTick() {
  if (!mediaStream || requestInFlight || cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  requestInFlight = true;
  try {
    const width = cameraPreview.videoWidth || 640;
    const height = cameraPreview.videoHeight || 480;
    captureCanvas.width = width;
    captureCanvas.height = height;

    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraPreview, 0, 0, width, height);

    const blob = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/png"));
    if (!blob) {
      throw new Error("Failed to capture a frame from the camera");
    }

    const payload = await postFrame(blob);
    applyPrediction(payload);
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
  if (inferenceTimer) {
    window.clearInterval(inferenceTimer);
    inferenceTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  cameraPreview.srcObject = null;
  drawHandLandmarks([]);
  setCameraUiState({
    pill: "idle",
    message: "Camera is off. Start the stream to run live inference from your browser.",
    running: false,
  });
}

async function startCameraStream() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    cameraPreview.srcObject = mediaStream;
    await cameraPreview.play();

    setCameraUiState({
      pill: "running",
      message: "Camera is live. Frames are being sampled and sent to the inference endpoint.",
      running: true,
    });

    inferenceTimer = window.setInterval(runInferenceTick, INFERENCE_INTERVAL_MS);
    await runInferenceTick();
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
    feature_dim: statTargets.feature_dim.textContent,
    hand_landmarks: [],
  });
  lastSpokenWordKey = "";
  stopSpeechPlayback();
  setSpeechUiState("idle", "Buffer reset. Waiting for a new finalized word.");
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
    cameraMessage.textContent = "Manual frame uploaded successfully.";
  } catch (error) {
    const message = error instanceof Error ? error.message : "Manual upload failed";
    setCameraUiState({
      pill: "error",
      message,
      running: Boolean(mediaStream),
    });
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

speakWordButton?.addEventListener("click", async () => {
  const textToSpeak = getSpeakText();
  setSpeechUiState(
    "clicked",
    textToSpeak ? `Speech request started for: ${textToSpeak}` : "Speech button clicked, but there is no finalized word or accepted buffer yet.",
  );
  setSpeakButtonBusy(true);
  try {
    const spoke = await speakWord(textToSpeak);
    if (spoke) {
      const currentWord = getSpeakText();
      const currentStatus = statTargets.final_word_status?.textContent ?? "";
      lastSpokenWordKey = currentWord ? `${currentWord}:${currentStatus}` : "";
    }
  } finally {
    setSpeakButtonBusy(false);
  }
});

window.addEventListener("beforeunload", () => {
  stopSpeechPlayback();
  stopCameraStream();
});

window.addEventListener("load", () => {
  preloadSpeechEngine().catch(() => {
    // The UI already shows the preload error; keep the app usable for inference.
  });
});
