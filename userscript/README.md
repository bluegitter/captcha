# 油猴自动识别验证码

`captcha-auto-fill.user.js` 监听页面对 `/api/code` 的 `fetch` 和 `XMLHttpRequest` 请求，取得验证码图片后使用 ONNX Runtime Web 推理，并把 5 位结果写入验证码输入框。

## 重要限制

浏览器不能直接加载 PyTorch 的 `.pt` 文件。因此当前地址：

```text
http://192.168.14.162:9090/model/captcha_crnn_a100_int8.pt
```

不能直接填入油猴脚本使用。需要先在 Python 环境导出 ONNX，并把 `.onnx` 文件放到 HTTP 服务目录：

可复用 `crnn_model.py` 中的 `CRNNFixedCaptcha` 定义，导出代码示例：

```python
import torch
from crnn_model import CRNNFixedCaptcha

ckpt = torch.load('artifacts/captcha_crnn_a100.pt', map_location='cpu')
model = CRNNFixedCaptcha(); model.load_state_dict(ckpt['state_dict']); model.eval()
torch.onnx.export(model, torch.zeros(1, 1, 38, 120),
                  'captcha_crnn_fp32.onnx', opset_version=17,
                  input_names=['image'], output_names=['logits'],
                  dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}})
```

然后使用 ONNX Runtime 的 `quantize_dynamic` 对 `MatMul`/`Gemm` 做动态 INT8 量化。

导出后，把脚本中的地址改为：

```javascript
modelUrl: 'http://192.168.14.162:9090/model/captcha_crnn_a100_int8.onnx' // 已部署模型
```

脚本通过 Tampermonkey 的 `GM_xmlhttpRequest` 下载模型，因此模型服务器不需要配置 CORS；需要保证该地址可以被当前浏览器访问。验证码接口仍需由目标页面自身正常请求。

## 安装和使用

1. 安装 Tampermonkey。
2. 新建脚本，把 `captcha-auto-fill.user.js` 全部粘贴进去并保存。
3. 确认 `@match` 覆盖目标页面 `http://192.168.63.70:7777/*`。
4. 打开或刷新目标页面。
5. 页面请求 `/api/code` 返回图片后，脚本会自动识别并填入输入框。

如果页面使用了不同的输入框名称，修改 `CONFIG.inputSelectors`。脚本默认支持 `code`、`captcha`、`validateCode`、验证码 placeholder、`#code` 和 `#captcha`。
