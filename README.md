# 验证码识别模型实验项目

本项目用于复现生产验证码的样本生成、模型训练、评估、压缩和浏览器部署流程。Java 生成器内置了生产端 Kaptcha 的关键配置，不依赖原始业务工程或 `cloud-common-captcha`。

## 项目结构

```text
.
├── captcha_model.py             # CNN 基线训练、测试、单图预测
├── crnn_model.py                # CNN + BiLSTM CRNN 训练和预测
├── compress_model.py            # PyTorch FP16/动态 INT8 压缩
├── export_onnx.py               # 导出 FP32 和浏览器 INT8 ONNX
├── generate_dataset.py          # Python 样本生成器
├── split_dataset.py             # 数据集划分
├── java/                        # 独立 Java Kaptcha 样本生成器
│   └── src/main/resources/
│       └── captcha.properties   # 生产验证码配置副本
├── userscript/                  # 油猴自动识别脚本
├── web/                         # ONNX Runtime Web 测试页面
├── docs/                        # 学习文档和实验记录
├── data/                        # 本地数据集，已忽略，不提交
└── artifacts/                   # 本地模型文件，已忽略，不提交
```

GitHub 只提交代码、配置模板、测试页面和文档。训练图片、模型权重、ONNX 工件和本地环境都由 `.gitignore` 排除，避免提交大文件和生产数据。

## 环境安装

需要 Python 3.10–3.13 和 Java 8+。Python 依赖：

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install -r requirements.txt
```

Java 生成器只使用 Maven Central 的 Kaptcha 依赖，不需要 `cloud` 目录或本机 `cloud-common-captcha` JAR。

## 生成和划分样本

Java 生成器使用 `java/src/main/resources/captcha.properties` 中的 120×38、5 位字符、字符集、字体、噪声和 WaterRipple 配置：

```bash
mvn -q -f java/pom.xml compile dependency:build-classpath \
  -Dmdep.outputFile=/tmp/captcha-java-classpath.txt

java -cp "java/target/classes:$(cat /tmp/captcha-java-classpath.txt)" \
  com.tellhow.captcha.CaptchaDatasetGenerator data/synthetic 10000

python split_dataset.py data/synthetic data/split --test-ratio 0.2 --seed 20261114
```

文件名包含标签，例如 `a8c5n_000001.jpg`。训练集和测试集应保持生成配置一致，并避免同一图片或增强版本跨集合泄漏。

## 训练和评估

CNN 基线：

```bash
python captcha_model.py train data/split/train \
  --epochs 20 --batch 128 --lr 0.0005 \
  --model artifacts/captcha_cnn_java_v2.pt
python captcha_model.py test data/split/test \
  --model artifacts/captcha_cnn_java_v2.pt
```

CRNN 模型：

```bash
python crnn_model.py train data/split/train \
  --epochs 20 --batch 512 --lr 0.0005 \
  --model artifacts/captcha_crnn_a100.pt
python crnn_model.py test data/split/test \
  --model artifacts/captcha_crnn_a100.pt
python crnn_model.py predict artifacts/captcha_crnn_a100.pt path/to/captcha.jpg
```

A100 实验结果：CNN 字符准确率 95.43%、完整验证码准确率 81.95%；CRNN 字符准确率 99.83%、完整验证码准确率 99.15%。完整验证码准确率要求 5 个字符全部正确，通常低于字符准确率。

## 模型压缩和 ONNX

```bash
python compress_model.py fp16 artifacts/captcha_crnn_a100.pt \
  artifacts/captcha_crnn_a100_fp16.pt
python compress_model.py int8 artifacts/captcha_crnn_a100.pt \
  artifacts/captcha_crnn_a100_int8.pt

python export_onnx.py \
  --model artifacts/captcha_crnn_a100.pt \
  --fp32 artifacts/captcha_crnn_a100.onnx \
  --int8 artifacts/captcha_crnn_a100_int8.onnx
```

浏览器版本只量化 `MatMul`/`Gemm`，避免部分 WASM 运行时不支持 `ConvInteger`。模型输入为 `[N,1,38,120]`，输出为 `[N,5,20]`。

## 浏览器测试

将 ONNX 文件放到 `web/models/` 后启动静态服务器：

```bash
cp artifacts/captcha_crnn_a100_int8.onnx web/models/
python -m http.server 8080 --directory web
```

打开 <http://localhost:8080/>，选择验证码图片并执行识别。不能直接用 `file://` 打开页面。

## 油猴脚本

脚本位于 `userscript/captcha-auto-fill.user.js`，支持以下登录页面：

```text
http://192.168.63.70:7777/*
http://192.168.63.56:9090/*
http://192.168.63.56:8090/*
```

脚本监听页面已经加载的验证码图片，直接读取图片内容识别，避免再次请求验证码接口导致图片刷新。当前模型地址配置在脚本中，部署到公开环境前应替换为可访问的模型地址。

## 文档

- [验证码特点与模型选型](docs/验证码特点与模型选型.md)
- [验证码识别模型训练指南](docs/验证码识别模型训练指南.md)
- [油猴脚本说明](userscript/README.md)
