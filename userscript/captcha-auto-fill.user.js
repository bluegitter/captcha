// ==UserScript==
// @name         Cloud CAPTCHA 自动识别填充
// @namespace    tellhow-cloud-captcha
// @version      1.2.0
// @description  拦截 /api/code 验证码图片，使用 ONNX Runtime Web 识别并填入验证码输入框
// @match        http://192.168.63.70:7777/*
// @match        https://192.168.63.70:7777/*
// @match        http://192.168.63.56:9090/*
// @match        https://192.168.63.56:9090/*
// @match        http://192.168.63.56:8090/*
// @match        https://192.168.63.56:8090/*
// @run-at       document-start
// @grant        GM_xmlhttpRequest
// @connect      192.168.14.162
// @require      https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js
// ==/UserScript==

(() => {
  'use strict';

  const CONFIG = {
    // 浏览器不能直接加载 .pt。请将 PyTorch 模型导出为 ONNX 后填写此地址。
    // 例如：http://192.168.14.162:9090/model/captcha_crnn_a100_int8.onnx
    modelUrl: 'http://192.168.14.162:9090/model/captcha_crnn_a100_int8.onnx',
    apiPath: '/api/code',
    imageSelector: 'img[alt="验证码"], img.login-code-img',
    width: 120,
    height: 38,
    chars: '2345678abcdefgmnpwxy',
    inputSelectors: [
      'input[name="code"]', 'input[name="captcha"]',
      'input[name="validateCode"]', 'input[placeholder*="验证码"]',
      'input[placeholder*="校验码"]', '#code', '#captcha'
    ]
  };

  // Tampermonkey 的 @require 可能把 UMD 全局放在沙箱变量而非 window.ort。
  const ORT = (() => {
    try { if (typeof ort !== 'undefined') return ort; } catch (_) { /* sandbox */ }
    return globalThis.ort || (typeof unsafeWindow !== 'undefined' ? unsafeWindow.ort : undefined);
  })();
  let sessionPromise;
  let lastImageUrl;

  function log(...args) { console.debug('[captcha-auto-fill]', ...args); }
  function notify(message, error = false) {
    const old = document.querySelector('#captcha-auto-fill-status');
    if (old) old.remove();
    const el = document.createElement('div');
    el.id = 'captcha-auto-fill-status';
    el.textContent = message;
    Object.assign(el.style, {
      position: 'fixed', zIndex: 2147483647, right: '12px', bottom: '12px',
      padding: '7px 10px', color: '#fff', background: error ? '#c62828' : '#2e7d32',
      borderRadius: '4px', font: '13px sans-serif'
    });
    (document.body || document.documentElement).appendChild(el);
    setTimeout(() => el.remove(), 4500);
  }

  function findInput() {
    for (const selector of CONFIG.inputSelectors) {
      const input = document.querySelector(selector);
      if (input && !input.disabled && !input.readOnly) return input;
    }
    return document.querySelector('input[type="text"]:not([disabled]):not([readonly])');
  }

  async function loadSession() {
    if (!ORT) throw new Error('ONNX Runtime Web 未加载');
    if (CONFIG.modelUrl.endsWith('.pt')) {
      throw new Error('浏览器不能直接加载 PyTorch .pt，请先导出 ONNX');
    }
    ORT.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    ORT.env.wasm.numThreads = 1;
    // 通过 GM_xmlhttpRequest 下载模型，避免模型服务器未配置 CORS 时被浏览器拦截。
    const bytes = await new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: 'GET', url: CONFIG.modelUrl, responseType: 'arraybuffer',
        onload: response => response.status >= 200 && response.status < 300
          ? resolve(new Uint8Array(response.response))
          : reject(new Error(`模型下载失败：HTTP ${response.status}`)),
        onerror: () => reject(new Error('模型下载失败：网络错误')),
        ontimeout: () => reject(new Error('模型下载超时'))
      });
    });
    return ORT.InferenceSession.create(bytes, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all'
    });
  }

  function imageTensor(blob) {
    return createImageBitmap(blob).then(bitmap => {
      const canvas = document.createElement('canvas');
      canvas.width = CONFIG.width; canvas.height = CONFIG.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0, CONFIG.width, CONFIG.height);
      bitmap.close();
      const pixels = ctx.getImageData(0, 0, CONFIG.width, CONFIG.height).data;
      const values = new Float32Array(CONFIG.width * CONFIG.height);
      for (let y = 0; y < CONFIG.height; y++) {
        for (let x = 0; x < CONFIG.width; x++) {
          const i = (y * CONFIG.width + x) * 4;
          values[y * CONFIG.width + x] =
            (0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]) / 255;
        }
      }
      return new ORT.Tensor('float32', values, [1, 1, CONFIG.height, CONFIG.width]);
    });
  }

  function decode(output) {
    const values = output.data;
    const classes = CONFIG.chars.length;
    let result = '';
    for (let position = 0; position < 5; position++) {
      let best = 0;
      for (let i = 1; i < classes; i++) {
        if (values[position * classes + i] > values[position * classes + best]) best = i;
      }
      result += CONFIG.chars[best];
    }
    return result;
  }

  async function recognize(blob) {
    sessionPromise ||= loadSession();
    const session = await sessionPromise;
    const tensor = await imageTensor(blob);
    const output = await session.run({ [session.inputNames[0]]: tensor });
    return decode(output[session.outputNames[0]]);
  }

  function fill(value) {
    const input = findInput();
    if (!input) { log('未找到验证码输入框'); return; }
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    input.dispatchEvent(new Event('blur', { bubbles: true }));
    notify(`验证码已填入：${value}`);
  }

  async function processResponse(response) {
    const url = new URL(response.url, location.href);
    if (!url.pathname.endsWith(CONFIG.apiPath)) return;
    const copy = response.clone();
    const blob = await copy.blob();
    if (!blob.type.startsWith('image/')) return;
    if (lastImageUrl) URL.revokeObjectURL(lastImageUrl);
    lastImageUrl = URL.createObjectURL(blob);
    try {
      notify('正在识别验证码…');
      fill(await recognize(blob));
    } catch (e) {
      console.error('[captcha-auto-fill]', e);
      notify(`识别失败：${e.message}`, true);
    }
  }

  async function processImageElement(image) {
    const src = image.currentSrc || image.src;
    if (!src || image.dataset.captchaProcessed === src) return;
    // DOM 插入时图片可能尚未下载完成，必须等待 load 后再绘制 Canvas。
    if (!image.complete || image.naturalWidth === 0) {
      if (image.dataset.captchaLoadWaiting !== src) {
        image.dataset.captchaLoadWaiting = src;
        image.addEventListener('load', () => processImageElement(image), { once: true });
      }
      return;
    }
    image.dataset.captchaProcessed = src;
    try {
      notify('正在识别验证码…');
      // 直接读取页面已经加载完成的图片，避免再次请求 /api/code 生成新验证码。
      const canvas = document.createElement('canvas');
      canvas.width = CONFIG.width; canvas.height = CONFIG.height;
      canvas.getContext('2d').drawImage(image, 0, 0, CONFIG.width, CONFIG.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('无法读取页面中的验证码图片');
      fill(await recognize(blob));
    } catch (e) {
      console.error('[captcha-auto-fill]', e);
      notify(`识别失败：${e.message}`, true);
    }
  }

  function watchCaptchaImages() {
    const scan = root => {
      if (root?.matches?.(CONFIG.imageSelector)) processImageElement(root);
      root?.querySelectorAll?.(CONFIG.imageSelector).forEach(processImageElement);
    };
    scan(document);
    new MutationObserver(mutations => {
      for (const mutation of mutations) {
        mutation.addedNodes.forEach(scan);
        if (mutation.type === 'attributes' && mutation.target.matches?.(CONFIG.imageSelector)) {
          processImageElement(mutation.target);
        }
      }
    }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['src'] });
    document.addEventListener('load', event => {
      if (event.target.matches?.(CONFIG.imageSelector)) processImageElement(event.target);
    }, true);
  }

  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);
    processResponse(response).catch(e => log(e));
    return response;
  };

  const open = XMLHttpRequest.prototype.open;
  const send = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__captchaUrl = new URL(url, location.href).pathname;
    return open.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    if (this.__captchaUrl && this.__captchaUrl.endsWith(CONFIG.apiPath)) {
      this.addEventListener('load', () => {
        if (this.response instanceof Blob) processResponse(new Response(this.response, { url: this.responseURL })).catch(log);
      });
    }
    return send.apply(this, args);
  };

  watchCaptchaImages();
  window.addEventListener('load', () => {
    if (CONFIG.modelUrl.endsWith('.pt')) notify('请先把 .pt 转换为 ONNX，油猴脚本才能在浏览器运行', true);
  });
})();
