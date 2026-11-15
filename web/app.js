// The package exposes its browser ESM entry under dist/esm; dist/ort.min.mjs
// does not exist on jsDelivr.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/esm/ort.min.js";

// Keep the runtime WASM binaries separate from the application origin. Without
// this setting ORT requests /ort-wasm-simd.wasm from localhost:8080.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
ort.env.wasm.numThreads = 1;

const CHARS = "2345678abcdefgmnpwxy";
const WIDTH = 120;
const HEIGHT = 38;
let session = null;
let image = null;

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };
const error = (text = "") => { $("error").textContent = text; };

async function loadModel() {
  error();
  const modelPath = $("model").value;
  status("加载中…");
  try {
    const webgpu = modelPath.includes("fp16");
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: webgpu ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all"
    });
    status(`已加载（${session.inputNames[0]} → ${session.outputNames[0]}）`);
    $("predict").disabled = !image;
  } catch (e) {
    session = null;
    status("加载失败");
    error(`模型加载失败：${e.message}\n请确认模型文件位于 public/models，并通过 HTTP 服务访问页面。`);
  }
}

function chooseFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    image = new Image();
    image.onload = () => {
      $("preview").src = reader.result;
      $("preview").hidden = false;
      $("predict").disabled = !session;
      error();
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function imageTensor() {
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH; canvas.height = HEIGHT;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, WIDTH, HEIGHT);
  const pixels = ctx.getImageData(0, 0, WIDTH, HEIGHT).data;
  const data = new Float32Array(WIDTH * HEIGHT);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = (y * WIDTH + x) * 4;
      data[y * WIDTH + x] = (0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]) / 255;
    }
  }
  return new ort.Tensor("float32", data, [1, 1, HEIGHT, WIDTH]);
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

async function predict() {
  if (!session || !image) return;
  error(); $("result").textContent = "识别中…";
  const started = performance.now();
  try {
    const output = await session.run({ [session.inputNames[0]]: imageTensor() });
    const tensor = output[session.outputNames[0]];
    const logits = tensor.data;
    const text = [];
    const confidence = [];
    for (let position = 0; position < 5; position++) {
      const offset = position * CHARS.length;
      const probabilities = softmax(Array.from(logits.slice(offset, offset + CHARS.length)));
      let best = 0;
      for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
      text.push(CHARS[best]); confidence.push((probabilities[best] * 100).toFixed(2) + "%");
    }
    $("result").textContent = text.join("");
    $("meta").textContent = `字符置信度：${confidence.join(" / ")}；耗时：${(performance.now() - started).toFixed(1)} ms`;
  } catch (e) {
    $("result").textContent = "";
    error(`识别失败：${e.message}`);
  }
}

$("load").addEventListener("click", loadModel);
$("predict").addEventListener("click", predict);
$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
$("drop").addEventListener("dragover", (e) => { e.preventDefault(); $("drop").classList.add("drag"); });
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("drag"));
$("drop").addEventListener("drop", (e) => { e.preventDefault(); $("drop").classList.remove("drag"); chooseFile(e.dataTransfer.files[0]); });
