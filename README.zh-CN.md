# toktier

[English](README.md) | **简体中文**

**整段对话只分词一次，之后只处理新增内容。**

toktier 是面向智能体 LLM 服务的有状态分词系统。它保留每个会话的 token
状态，用经过认证的 CPU 路径修复追加内容，并为新请求或大请求提供经过认证的
GPU 路径。两条快速路径返回的 token ID 都与 Hugging Face（HF）`tokenizers`
从头完整编码的结果**逐位一致**。

- **大规模精确验证。** 发布验证覆盖 14 个 tokenizer 工件、38 亿篇真实文档，
  共记录 **532 亿次检查**（12.33 万亿字符），未观察到任何分歧。
- **CPU 与 GPU 两条快路径。** 在归档基准中，GPU 路径处理一个全新的
  400 万字符请求（约 78.6 万 token）仅需 **3.88 ms**；CPU 修复路径向
  419 万字符会话追加 256 个字符仅需 **1.68 ms**。
- **先认证，再加速。** 只有 tokenizer 工件、参考版本、kernel 交付形式和
  GPU 架构都落在证据覆盖范围内时，系统才会采用快速路径。`explain()` 会
  说明实际路线及其原因。

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="docs/figures/hero_session_vs_reencode_dark.svg">
  <img alt="toktier 与完整重编码在三种 400 万字符会话负载下的延迟对比，线性坐标"
       src="docs/figures/hero_session_vs_reencode.svg">
</picture>

图中每根柱都是实测中位数。精确数值、负载大小和样本数见
[`hero_session_vs_reencode.data.json`](docs/figures/hero_session_vs_reencode.data.json)，
完整扫描结果见 [`docs/benchmarks.md`](docs/benchmarks.md)。

## 快速开始

运行 `pip install toktier` 即可安装；GPU 选项见[安装](#安装)。

```python
import toktier

tok = toktier.load("qwen3_8b")          # support matrix 中的 family id
enc = tok.encode("hello world")         # token ID
print(enc.ids)
print(tok.decode(enc.ids))
print(tok.explain())                    # 实际后端及选择原因
```

为持续增长的对话指定 `session=`，即可跨调用、跨进程保存 token 状态：

```python
tok = toktier.load("qwen3_8b", store="./toktier-store")

transcript = "user: hello\nassistant: hello! how can I help?\n"
enc = tok.encode(transcript, session="chat-42")

transcript += "user: what changed since my last call?\n"
enc = tok.encode(transcript, session="chat-42")

# 存储路径与从头编码路径返回相同的 ID。
assert enc.ids == tok.encode(transcript, lookup="off").ids
```

不传 `session=` 时，store 也可以按内容查找经过逐字节验证的已存前缀。
传入 `lookup="off"` 可跳过查找。字节验证失败只会成为一次 miss，不会被当作
可信命中；缓存淘汰只影响延迟，不影响输出。

路由策略可以显式选择，并且全程可检查：

```python
from toktier import RoutingPolicy

tok = toktier.load("qwen3_8b", policy=RoutingPolicy.CERTIFIED)
```

| Policy | 采用的路径 | 快速路径前提不成立时 |
|---|---|---|
| `CERTIFIED`（默认） | 只采用精确工件、HF 版本、引擎/kernel 字节、交付形式和硬件均有证据覆盖的路径 | 回退 HF，并记录原因 |
| `REFERENCE` | 只使用 HF `tokenizers` | 不尝试任何加速路径 |
| `REQUIRE_ACCELERATED` | 与 `CERTIFIED` 相同的认证路径 | 若构造时没有合格快速路径则报错；逐输入安全回退仍然保留 |
| `EXPERIMENTAL` | 评估时可以采用未经判决的组合 | 明确标出每个被豁免的前提；永远不是默认策略 |

在默认 `CERTIFIED` 策略下，安装 profile 与输入形态共同决定自动路线：

| 情形 | 自动路线 |
|---|---|
| `toktier`，11 个认证 tokenizer 工件（覆盖 12 个模型家族）之一 | 修正版 Gigatoken 完成 CPU 全量编码；任一绑定检查失败则用 HF |
| `toktier[gpu]`，小于 64 KiB 的冷请求/普通请求 | 修正版 Gigatoken CPU 路径（没有 CPU-fast 认证的家族使用 HF） |
| `toktier[gpu]`，至少 64 KiB 的冷请求/普通请求 | 随包预编译 GPU；失败时按冻结链条依次转到修正版 Gigatoken、HF |
| 已存在的 session 收到严格追加 | 12 个覆盖家族使用修正版 Gigatoken CPU repair，不受完整对话总长度影响 |
| Added-token 或 repair guard 无法证明前提 | 该输入使用 HF 参考路径 |

`explain()` 会报告冻结的后端链、64 KiB 判定、最后实际返回结果的后端，以及
每类 fallback 计数。

## 安装

```bash
pip install toktier                 # 完整的认证 CPU 产品
pip install "toktier[gpu]"          # CPU 产品 + 自动预编译 GPU 路由
pip install "toktier[gpu-jit]"      # 相同路由，本机 JIT 交付
```

| 安装项 | 交付内容 | 要求 |
|---|---|---|
| `toktier` | 修正版 Gigatoken CPU 全量编码与 session repair、HF fallback、持久化 store、路由和 CLI | Linux x86_64、glibc 2.34+、CPython 3.10+；固定安装 `tokenizers==0.22.2` 与 `transformers==4.57.6` |
| `toktier[gpu]` | `toktier` 的严格超集；按 64 KiB 阈值自动路由到随包多架构 CUDA fatbin | NVIDIA GPU、580.65.06+ 驱动、`torch`；无需编译器，首次使用不编译 |
| `toktier[gpu-jit]` | CPU/GPU 路由与 `toktier[gpu]` 相同；在本机编译认证 kernel 源码 | 已判决 CUDA/PyTorch 工具链、`nvcc`、`torch`、`ninja`；首次使用需要编译 |

JIT 在工具链边界严格 fail-closed。本机 CUDA/PyTorch 组合不在注册表的已判决
组合中时，自动路由会给出醒目 warning，并继续使用修正版 Gigatoken → HF
回退链；显式请求 CUDA 则会直接失败，同时列出本机组合、认证约束和可复制的
处理命令。已判决组合可提前编译：

```bash
toktier gpu compile qwen3_8b
```

如果只是评估未经判决的组合，必须显式接受风险：

```bash
toktier gpu compile qwen3_8b --accept-uncertified-jit
```

**这不会让生成的 kernel 获得认证。** 该命令使用 `EXPERIMENTAL` 策略，打印
`UNCERTIFIED JIT OPT-IN` 警告，并记录所有被豁免的前提。应用程序仍须逐进程
显式传入 `policy="experimental", gpu_delivery="jit"`；风险接受不会持久化，也
不会被后续认证进程继承。使用结果前请检查
`explain()["experimental_waivers"]`。

经过修正并固定 Unicode 数据版本的 Gigatoken 原生模块已经随 Core wheel 交付，
其私有导入名为 `toktier._vendor.gigatoken_rs`。TokTier 不会安装或信任顶层
`gigatoken` 同名包。基础 wheel 也固定安装打开该认证路径所需的 HF loader 与
oracle 版本，不再需要单独安装 CPU-fast 组件。

如需核对来源，可在源码 checkout 中复现相同的原生字节：

```bash
pip install .
TOKTIER_GIGATOKEN_BUILD_ROOT="$PWD/.build/gigatoken" \
  packaging/fast_cpu/build_pinned.sh
```

[可复现构建配方](packaging/fast_cpu/README.md)固定上游 commit、补丁、Unicode
数据、编译器和构建后端；其输出只用于复现与审计，不是另一个运行时安装项。
运行时注册表会核对随包原生模块摘要、repair 配置、oracle 与 tokenizer 工件。
Core wheel 同时携带 Gigatoken 的 MIT 许可证、TokTier 修改声明、依赖 SBOM 和
完整依赖许可证束。

0.1.0 只发布 ABI3 Linux x86-64 wheel，不发布 sdist：认证 CPU 二进制要求
glibc 2.34 或更新版本，而在 sdist 安装阶段静默重编译会产生不同、未认证的
字节。对应 git tag 包含完整源码与固定版本的复现配方。

预编译 fatbin 包含 `sm_75/80/86/89/90/100/120` 镜像以及
`compute_75` PTX fallback。绑定二进制摘要的认证覆盖 `sm_89` 和
`sm_120`；其他随包架构标记为 `experimental`。默认 facade 会为
`toktier[gpu]` 惰性选择预编译交付、为 `toktier[gpu-jit]` 惰性选择 JIT；显式
`gpu_delivery=` 参数可覆盖 profile 识别。JIT 交付在 `sm_89` 和
`sm_120` 上的状态为 `certified_source`，其证书绑定源码、class table、
编译参数和工具链约束，而不是每台机器各自生成的二进制。自动 facade、显式
GPU API 和交付诊断见 [`docs/gpu-jit.md`](docs/gpu-jit.md)。

Tokenizer 工件不打包在 wheel 中，而是从固定的上游 revision 获取并用
SHA-256 验证。CLI 同时支持联网、镜像和气隙环境：

```bash
toktier artifacts fetch qwen3_8b
toktier artifacts export qwen3_8b --out qwen3_8b.tar
toktier artifacts import qwen3_8b.tar
toktier artifacts verify qwen3_8b
toktier inspect qwen3_8b
toktier doctor --json
```

核心包不依赖 `torch`，导入时不需要 CUDA、网络连接或硬件探测。工件缓存、
编译 kernel 缓存和持久化会话状态分别存放。

Fastokens 0.3.1 只作为显式实验性对照使用：

```python
tok = toktier.load(
    "qwen3_8b", policy="experimental", repair_backend="fastokens"
)
```

该适配器会重新编码完整会话，并报告 `exact_id_guarantee: false`；认证策略永远
不会自动选择它，它也不属于修正版 Gigatoken 的 12.4 TB 验证声明。

## 正确性与证据

认证参考实现是默认配置下的 Hugging Face `tokenizers` 0.22.2。认证绑定
精确的工件字节和参考版本。如果本机 HF 版本不在认证集合内，加速路由会被
关闭，请求继续使用本机安装的参考路径。

| 验证活动 | 规模 | 记录到的分歧 |
|---|---:|---:|
| 全语料差分验证 | 14 个工件 × 3,800,016,491 篇文档 = **53,200,230,874 次检查** | 0 |
| 语料体量 | 12,328,592,579,973 个 Unicode code point | — |
| 发布代码路径一致性验证 | 15,960,166 篇文档 | 0 |
| 修正版 Gigatoken CPU repair | 11 个唯一工件 × 3,800,016,491 篇文档 = **41,800,181,401 次检查**（通过工件逐字节相同覆盖 12 个模型家族） | 0 |

机器可读记录位于
[`evidence/evidence_manifest.json`](evidence/evidence_manifest.json)、
[`evidence/evidence_manifest_added_families.json`](evidence/evidence_manifest_added_families.json)
和 [`tables/support_registry.json`](tables/support_registry.json)。随包的
逐工件读数覆盖 49,920,199,013 次检查；早期归档阶段覆盖其余
3,280,031,861 次，两者相加得到上面的总数。另有一轮覆盖全部 11 个 CPU
快速工件、直接通过发布版公开会话 API 运行的定点复验，记录在
[`readings/fast_cpu_focused_parity.json`](readings/fast_cpu_focused_parity.json)。

三个状态把证据与运行时行为区分开：

| 状态 | 含义 |
|---|---|
| `certified` | 证据绑定精确工件与加速二进制。 |
| `certified_source` | 对本机编译，证据绑定源码、构建输入和约束。 |
| `reference-only` | 不采用加速路径，运行 HF `tokenizers`。 |

这些结果是大规模经验性差分验证，不是对所有可能输入的数学证明。因此，
逐请求检查和参考 fallback 仍然是系统契约的一部分。

仓库自检命令：

```bash
python tools/generate_evidence.py --check
python tools/validate_registry.py
python tools/generate_registry.py --release-check
python tools/dev.py test-packaging
```

## 性能

README 顶部的图比较了自动 GPU/repair 路由与相同文本上的完整重编码。发布的
批量 GPU 路径还有一组同机吞吐数据：

| 路径 | 吞吐 | 环境 |
|---|---:|---|
| GPU 端到端（文本输入、ID 输出） | 0.6028 GB/s | 单张 RTX PRO 6000 Blackwell |
| HF 参考 CPU 路径 | 0.0047 GB/s | 同一主机、同一输入、单 CPU 核 |

这组数据使用 2.2 GB 驻留内存的真实网页文本，按墙钟时间统计 UTF-8 字节，
并包含主机侧 ID 数组物化。完整协议、全部数据格和 provenance 见
[`docs/benchmarks.md`](docs/benchmarks.md)。

主要实验使用 RTX PRO 6000 Blackwell，但相同协议下的消费级 RTX 5090 扫描
反而**快 11–17%**（已报告家族为 4.24–5.50 GB/s）。因此消费级硬件不是降级
模式：RTX 4090 也通过了 `sm_89` 正确性与预编译交付电池。这里并不承诺每张
消费卡都有相同比例的提升；实际性能仍取决于架构、负载和主机侧交付。

![单请求延迟](docs/figures/f1_single_request_latency.svg)

![会话尾延迟](docs/figures/f2_session_tail_latency.svg)

![会话状态内存](docs/figures/f3_session_state_memory.svg)

![repair 路径等效吞吐](docs/figures/f4_repair_equivalent_throughput.svg)

图中已明确标注 Hugging Face（HF）`tokenizers`，并为每幅图提供
`docs/figures/*.data.json` 机器可读数据。基准文档也保留了其他引擎直接使用
时更快的区间。

## 支持矩阵

| 路线 | 家族数 | 覆盖情况 |
|---|---:|---|
| 认证 CPU fast repair | 12 个模型家族 / 11 个唯一 tokenizer 工件 | 修正版 Gigatoken，12.33 万亿字符，未观察到 ID 分歧 |
| Byte-level BPE | 15 | CPU 证据；逐工件记录 GPU 状态 |
| WordPiece | 3 | CPU 证据 |
| 结构性排除 | 2 | 记录具体原因 |

[`docs/support-matrix.md`](docs/support-matrix.md) 列出了每个 anchor 工件、
SHA-256、后端状态，以及 **210 个已验证模型仓库**；这些仓库使用完全相同或
仅序列化形式不同的 tokenizer。覆盖关系由 tokenizer 内容决定，而不是仓库名。

当前 wheel 可以解析生成清单中的 14 个工件。`kimi_k3` 和三个 WordPiece
家族已有证据，但尚未接入该清单；`toktier inspect` 是随包可用列表的权威来源。

## 与现有工作的关系

Incremental BPE 研究关注 merge 阶段如何随新字节扩展。toktier 工作在它的
上一层：会话状态保存完整 tokenizer pipeline 的 token ID 与 span，包括
normalization、pre-tokenization、merge 和 added-token 处理；只有边界检查通过
后才接受一次追加。

`llm-tokenizer` 和 NVIDIA Dynamo 的 `dynamo-tokenizers` 等 serving 项目也会
缓存编码结果。主要接口差异如下：

| 属性 | toktier | 进程内前缀缓存 |
|---|---|---|
| 生命周期 | 可持久化、跨进程 | 跟随 tokenizer 进程 |
| 命中验证 | 摘要提出候选，存储字节最终验证 | 按摘要查找 |
| 复用边界 | 经过认证的 tokenizer 边界 | 通常是 special-token 边界 |
| 使用界面 | 面向自行管理会话的应用与智能体循环的 Python 库 | serving gateway 组件 |

两层共同使用的方法见
[`docs/integration/dynamo.md`](docs/integration/dynamo.md)。

## 文档

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 分层、路由与 store 格式。
- [`ROADMAP.md`](ROADMAP.md) — 发布范围与后续集成。
- [`docs/support-matrix.md`](docs/support-matrix.md) — 工件与覆盖仓库。
- [`docs/gpu-jit.md`](docs/gpu-jit.md) — 预编译和 JIT GPU 交付。
- [`docs/integration/dynamo.md`](docs/integration/dynamo.md) — Dynamo 集成。
- [`docs/paper/toktier-preprint.pdf`](docs/paper/toktier-preprint.pdf) — 最新论文预印本。

## 致谢

TokTier 的 CPU Fast Pass 和 Fast Repair 建立在
[Gigatoken](https://github.com/marcelroed/gigatoken) 与
[Fastokens](https://github.com/crusoecloud/fastokens) 两项优秀的开源工作之上。感谢
两项工作的作者和贡献者将这些成果开源。

修正版 Gigatoken 是 11 个唯一 tokenizer 工件的默认认证 repair-window 引擎；
由于 NVIDIA Nemotron-Terminal 逐字节复用 `qwen3_8b` tokenizer，这些工件覆盖
12 个模型家族。TokTier 的兼容性补丁使其 Unicode 数据和 UTF-8 处理与冻结的
[Hugging Face tokenizers](https://github.com/huggingface/tokenizers) 参考实现对齐；
该路径在 12.33 万亿字符上完成 418 亿次检查，未观察到 token ID 分歧。

Fastokens 0.3.1 保留为显式实验性备选；TokTier 不对该适配器承诺逐 ID 一致，也
不会自动选择它。Fastokens 采用 Apache-2.0，Gigatoken 采用 MIT；准确 revision、
许可证副本、Gigatoken 补丁与修改声明见
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) 和 [`packaging/`](packaging/)。

## 许可证与引用

toktier 采用 [Apache License 2.0](LICENSE)；署名信息见
[`NOTICE`](NOTICE)。

**论文：** [*TokTier: Exact Stateful CPU+GPU Tokenization for Agentic LLM
Serving*](https://arxiv.org/abs/2607.29678) ·
[PDF](https://arxiv.org/pdf/2607.29678)

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{zhang2026toktier,
  title         = {TokTier: Exact Stateful CPU+GPU Tokenization for Agentic LLM Serving},
  author        = {Zhenyu Zhang and Zhichao Cao},
  year          = {2026},
  eprint        = {2607.29678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2607.29678},
  url           = {https://arxiv.org/abs/2607.29678}
}
```

机器可读的引用元数据见 [`CITATION.cff`](CITATION.cff)。
