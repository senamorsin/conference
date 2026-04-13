const cameraPreview = document.getElementById("camera-preview");
const captureCanvas = document.getElementById("capture-canvas");
const landmarkOverlay = document.getElementById("landmark-overlay");
const cameraStatus = document.getElementById("camera-status");
const cameraMessage = document.getElementById("camera-message");
const startCameraButton = document.getElementById("start-camera");
const stopCameraButton = document.getElementById("stop-camera");
const resetStateButton = document.getElementById("reset-state");
const uploadForm = document.getElementById("upload-form");

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

window.addEventListener("beforeunload", () => {
  stopCameraStream();
});
