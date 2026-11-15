# CAPTCHA recognition experiment

`CodeEndpoint` uses Google Kaptcha `Producer.createText()` and `createImage(text)`, then stores the text in Redis. The Java generator under `java/` calls the same `cloud-common-captcha` component and writes the generated text into each image filename.

```bash
# Generate with the same Java Kaptcha configuration as CodeEndpoint.
mvn -f java/pom.xml compile dependency:build-classpath -Dmdep.outputFile=/tmp/captcha-java-classpath.txt
java -cp "java/target/classes:$(cat /tmp/captcha-java-classpath.txt)" \
  com.tellhow.captcha.CaptchaDatasetGenerator data/synthetic 3000

python captcha_model.py train data/synthetic --epochs 8

# Keep the test set separate from training and evaluate it after training.
python split_dataset.py data/synthetic data/split
python captcha_model.py train data/split/train --epochs 8 \
  --model artifacts/captcha_cnn_java.pt
python captcha_model.py test data/split/test \
  --model artifacts/captcha_cnn_java.pt
python captcha_model.py predict artifacts/captcha_cnn.pt data/synthetic/$(find data/synthetic -name '*.png' | head -1 | xargs -n1 basename)
```

Labels are five lowercase alphanumeric characters in the filename (`a8c5n_000001.png`). Training reports character accuracy and complete CAPTCHA accuracy; the latter requires all five positions to be correct.

The Java Kaptcha component's default alphabet is `2345678abcdefgmnpwxy`; the model uses this exact 20-class vocabulary. With 10,000 Java-generated images (8,000 train / 2,000 test), the current test run reached 95.43% character accuracy and 81.95% complete-CAPTCHA accuracy.

`crnn_model.py` contains the CNN+BiLSTM comparison model. On the same split after 5 epochs (the recurrent model is substantially slower on MPS), it reached 28.90% character accuracy and 0.05% complete-CAPTCHA accuracy, below the CNN baseline. It needs a longer, better-tuned run before judging the architecture; the result demonstrates that replacing the head alone does not improve accuracy under a reduced training budget.

Using the existing server environment `/data1/anaconda3/envs/hsd` (PyTorch 2.6.0+cu124) and `CUDA_VISIBLE_DEVICES=3` on an A100, a 20-epoch run reached 99.83% character accuracy and 99.15% complete-CAPTCHA accuracy on the same 2,000-image test set. The copied checkpoint is `artifacts/captcha_crnn_a100.pt`.

`compress_model.py` exports compact checkpoints. Dynamic INT8 is recommended for CPU inference; it reduced the checkpoint from 21 MB to 5.9 MB while retaining 99.83% character and 99.15% complete-CAPTCHA accuracy. FP16 is recommended for GPU inference; it reduced the checkpoint to 10 MB with the same test metrics.

```bash
python compress_model.py int8 \
  artifacts/captcha_crnn_a100.pt \
  artifacts/captcha_crnn_a100_int8.pt
python compress_model.py test \
  artifacts/captcha_crnn_a100_int8.pt \
  data/split/test
```

The browser test page is in `web/`. Put `captcha_crnn_int8.onnx` (or the FP16/FP32 ONNX variant) under `web/models/`, then serve the directory over HTTP:

```bash
mkdir -p web/models
cp artifacts/captcha_crnn_int8.onnx web/models/
cd web && python -m http.server 8080
```

Open `http://localhost:8080/`, load the model, choose a JPG/PNG CAPTCHA, and click **识别**. Opening `index.html` directly with `file://` is not supported because browser module and model requests require HTTP.

For the browser INT8 model, quantize only `MatMul`/`Gemm` operators. Full dynamic quantization can produce `ConvInteger`, which is not implemented by every ONNX Runtime Web WASM build. The generated model files are local deployment artifacts and are ignored by Git.

Install `requirements.txt` with Python 3.10–3.13 before training. The current macOS Intel image has Python 3.14 and no compatible PyTorch wheel, so this checkout validation covers data generation and Python syntax; the training command is ready to run in the documented environment.

## 目录结构

```text
.
├── captcha_model.py                 # CNN 基线训练、测试和单图预测
├── crnn_model.py                    # CNN + BiLSTM CRNN 训练和预测
├── compress_model.py                # FP16/动态 INT8 PyTorch 压缩
├── export_onnx.py                   # 导出 FP32/浏览器 INT8 ONNX
├── generate_dataset.py              # Python 样本生成器
├── split_dataset.py                 # 训练集/测试集划分
├── requirements.txt                 # Python 依赖
├── java/                            # 使用生产 Kaptcha 配置生成样本
├── userscript/                      # 油猴自动识别脚本和说明
├── web/                             # ONNX Runtime Web 测试页面
├── 验证码识别模型训练指南.md          # 完整学习和实验记录
├── data/                            # 本地数据集，不提交到 Git
└── artifacts/                       # 本地模型工件，不提交到 Git
```

`data/`、`artifacts/`、`web/models/` 和 Python 缓存已经通过 `.gitignore` 排除。GitHub 仓库应提交训练代码、Java 生成器、浏览器页面、油猴脚本和文档；模型文件和生成图片建议通过制品仓库或内部 HTTP 地址分发。

当前油猴脚本使用的 ONNX 模型地址为：

```text
http://192.168.14.162:9090/model/captcha_crnn_a100_int8.onnx
```

该地址是内网地址。公开 GitHub 前请改成可访问的制品地址，或在脚本配置中保留为部署环境变量，避免依赖内网服务。
