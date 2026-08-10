# toktier

[English](README.md) | **简体中文**

**整段对话只分词一次，之后只处理新增内容。**

toktier 是面向智能体 LLM 服务的有状态分词系统。它保留每个会话的 token
状态，通过认证 CPU 路径对追加文本执行 repair，并为全新请求或大型请求提供
认证 GPU 路径。两条快速路径返回的 token ID 序列，都与 Hugging Face（HF）
`tokenizers` 从头完整编码得到的 token ID 序列**完全一致（bit-identical）**。

- **大规模精确验证。** 发布验证覆盖 14 个 tokenizer 工件、38 亿篇真实文档，
  共记录 **532 亿次检查**（12.33 万亿字符），未观察到任何不一致。
- **CPU 与 GPU 均有快速路径。** 在已记录的整套基准测试中，GPU 路径处理一个
  全新的 400 万字符请求（约 78.6 万 token）仅需 **3.88 ms**；对 419 万字符会话
  追加 256 个字符时，原生有界 CPU repair 仅需 **1.68 ms**。该读数对应
  `toktier repair (HF tokenizers window)` 这一测试系列，也就是该数据单元测量的
  repair 窗口；修正版 Gigatoken 窗口是同一图中的另一测试系列（追加 65,536 个
  字符时为 2.39 ms）。两者都是下方路由表涵盖的有界 repair；图表数据标明了
  每根柱所属的测试系列。上述测量仅涵盖 repair 操作本身，不包含将完整历史
  token 序列物化为 Python tuple 的成本；原生 Rust serving 集成可以保留会话
  状态并仅使用 repair 后的 token 后缀，从而避免物化完整历史序列。
- **先认证，再加速。** 只有精确的 tokenizer 工件、参考实现版本、kernel 交付方式
  和架构均有记录证据覆盖时，系统才会采用快速路径。`explain()` 会报告实际路由
  及其原因。

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="docs/figures/hero_session_vs_reencode_dark.svg">
  <img alt="toktier 与完整重编码在三种 400 万字符会话负载下的延迟对比，线性坐标"
       src="docs/figures/hero_session_vs_reencode.svg">
</picture>

图中每根柱都是实测中位数。精确数值、负载大小和样本数见
[`hero_session_vs_reencode.data.json`](docs/figures/hero_session_vs_reencode.data.json)，
完整的扫描测试结果见 [`docs/benchmarks.md`](docs/benchmarks.md)。

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

先 `encode` 再 `decode`，不一定能还原为完全相同的文本：如果 tokenizer 的处理
流程包含归一化（例如 NFC），返回的会是归一化后的文本。这是 tokenizer 自身的
行为，并非 TokTier 造成的不一致。TokTier 只保证 ID 一致，这项保证不受影响：
对于同一输入，其 ID 与 HF 从头编码的结果相同，而且两个解码器返回的文本也相同。

如果应用代码拿到的是 Hugging Face 模型仓库，而不是 TokTier family id，也可以
按 tokenizer 内容解析：

```python
tok = toktier.from_pretrained("Qwen/Qwen3-0.6B")
```

对于已登记的 sibling 或 canonical 仓库，`from_pretrained()` 会下载经审计的
不可变 revision（未知仓库在未显式传入 `revision=` 时解析 `main`），精确计算
实际文件的 SHA-256，再查询受根摘要校验的 210 仓库 sibling 注册表。字节完全相同、
canonical 化等价或序列化等价的记录，会通过同一套 CPU/GPU 路由
使用已经认证的 canonical 工件。已知仓库的字节内容一旦变化，或遇到任何未登记
内容，在允许参考实现 fallback 的策略下都会继续使用 HF；
`REQUIRE_ACCELERATED` 策略则会报错。`explain()["model_resolution"]` 会同时报告
来源身份和实际执行的 canonical 身份。`load(family)` 仍是直接的 family API，
也是适合气隙环境的路径。

为不断增长的对话记录指定 `session=`，即可跨调用和进程保存 token 状态：

```python
tok = toktier.load("qwen3_8b", store="./toktier-store")

transcript = "user: hello\nassistant: hello! how can I help?\n"
enc = tok.encode(transcript, session="chat-42")

transcript += "user: what changed since my last call?\n"
enc = tok.encode(transcript, session="chat-42")

# 存储路径与从头编码路径返回相同的 ID。
assert enc.ids == tok.encode(transcript, lookup="off").ids
```

不传 `session=` 时，store 也可以按内容找到经过逐字节验证的已存前缀。传入
`lookup="off"` 可跳过这次 lookup。字节验证失败只会被视为未命中，绝不会成为
可信命中；缓存淘汰只影响延迟，不影响输出。长会话的稳定前缀封存后，重启时
仍可复用：恢复记录前，TokTier 会先将它与调用方提供的历史前缀绑定；绑定缺失
或损坏时则执行冷编码。

路由策略可以选择，也可以查看：

```python
from toktier import RoutingPolicy

tok = toktier.load("qwen3_8b", policy=RoutingPolicy.CERTIFIED)
```

| 策略 | 执行的路由 | 快速路径前提不成立时 |
|---|---|---|
| `CERTIFIED`（默认） | 只执行精确工件、HF 版本、引擎/kernel 字节、交付方式和硬件均有证据覆盖的路由 | fallback 到 HF，并记录原因 |
| `REFERENCE` | 只使用 HF `tokenizers` | 不尝试任何加速路径 |
| `REQUIRE_ACCELERATED` | 与 `CERTIFIED` 相同的认证路由 | 若构造时没有合格快速路径则报错；仍启用针对单个输入的安全 fallback |
| `EXPERIMENTAL` | 评估时可以采用未经评审的组合 | 明确标出每项获豁免的前提；永远不是默认策略 |

在此基础上，安装配置和输入形态会决定自动路由：

| 默认 `CERTIFIED` 策略下的情形 | 自动路由 |
|---|---|
| `toktier`，11 个认证 tokenizer 工件（覆盖 12 个 family）之一 | 修正版 Gigatoken 完成 CPU 全量编码；任一绑定检查失败则使用 HF |
| `toktier[gpu]`，小于 GPU crossover（默认 64 KiB）的冷请求/普通请求 | 修正版 Gigatoken CPU 路径（没有 CPU-fast 认证的 family 使用 HF） |
| `toktier[gpu]`，达到或超过 GPU crossover（默认 64 KiB）的冷请求/普通请求 | 随包预编译 GPU 路径；随后按固定 fallback 链依次使用修正版 Gigatoken 和 HF |
| 已存在的 session 收到严格追加 | 覆盖范围内的 12 个 family 使用修正版 Gigatoken CPU repair，不受完整对话总长度影响 |
| Added-token 或 repair guard 无法证明前提 | 该输入使用 HF 参考路径 |

`explain()` 会报告固定路由链、crossover 判定（`gpu_min_bytes`，默认 64 KiB）、
最后实际返回结果的后端，以及每类 fallback 的计数。

## Rust serving API

工作区现在提供无需 Python 的 Rust serving facade，供直接保留 token 状态的
前端使用。它支持固定版本工件的获取、镜像同步和气隙操作，参考实现、修正版 CPU、
预编译或 direct-JIT GPU 路由，连续 token 缓冲区，不依赖 executor 的有界批处理，
持久化命名 session，以及原生增量 `TokenPatch` 结果：

```rust,no_run
use toktier::{Device, Runtime};

let runtime = Runtime::builder().device(Device::Auto).build()?;
let tokenizer = runtime.load("qwen3_8b")?;
let mut session = tokenizer.open_session("agent-42")?;
let seed = session.seed("user: hello\n")?;
let patch = session.append("assistant: hi\n")?;
# Ok::<(), toktier::Error>(())
```

`patch.keep_tokens()` 给出下游已保留 ID 缓冲区的截断位置；
`patch.replacement_ids()` 是 repair 后的精确后缀。除非调用方显式请求
`snapshot()`，此次追加不会分配完整历史 ID 序列。该 crate 自 0.2.0 起发布在
crates.io 上，并跟随包版本号，因此 `cargo add toktier` 会从注册表解析它。
Rust serving 接口见 [`docs/rust-api.md`](docs/rust-api.md)；工件获取、JIT、
并发与可复现的离线分发见 [`docs/rust-lifecycle.md`](docs/rust-lifecycle.md)。

从 0.1.1 开始，UTF-8 crossover 与 added-token 未命中预筛会在一次零分配的
Rust selector 调用中完成。在已记录的 RTX 5090 主机上，4M-byte 控制平面微基准
由 2.97 ms 降至 0.052 ms（57.5 倍）；该数据只衡量路由，与 tokenization 和
Python 返回值物化分开。详见
[`docs/native-routing.md`](docs/native-routing.md)。

## 安装

```bash
pip install toktier                 # 完整的认证 CPU 产品
pip install "toktier[gpu]"          # CPU 产品 + 自动预编译 GPU 路由
pip install "toktier[gpu-jit]"      # 相同路由，本机 JIT 交付
cargo add toktier                   # 无 Python 依赖的 Rust serving API
```

| 安装项 | 交付内容 | 要求 |
|---|---|---|
| `toktier` | 修正版 Gigatoken CPU 全量编码与 session repair、HF fallback、持久化 store、路由和 CLI | Linux x86_64、glibc 2.34+、CPython 3.10+；固定安装 `tokenizers==0.22.2` 与 `transformers==4.57.6` |
| `toktier[gpu]` | `toktier` 的严格超集；通过 64 KiB crossover 自动路由到随包的多架构 CUDA fatbin | NVIDIA GPU、驱动版本 580.65.06+、`torch`；无需编译器，首次使用不编译 |
| `toktier[gpu-jit]` | CPU/GPU 路由与 `toktier[gpu]` 相同；在本机编译认证 kernel 源码 | 经评审的 NVCC / torch-runtime CUDA / PyTorch 三元组、`torch`、`ninja`；首次使用需要编译 |

JIT 在工具链边界采用严格的 fail-closed 策略。认证会将 PyTorch 扩展构建器
实际选择的 `nvcc`、`torch.version.cuda` 和 PyTorch 发行版版本分别作为独立维度
核对。如果注册表未记录这个精确三元组，自动路由会给出醒目警告，并继续使用
修正版 Gigatoken → HF fallback 链；显式请求 CUDA 则会直接失败，同时列出观测到的
编译器/运行时三元组、认证约束和可直接复制的处理命令。例如，torch CUDA 13.0
配 NVCC 13.2 不会被视为经评审的 NVCC 13.0 组合。经评审的组合可在首次使用前编译：

```bash
toktier gpu compile qwen3_8b
```

如果只是评估未经评审的组合，必须显式确认风险：

```bash
toktier gpu compile qwen3_8b --accept-uncertified-jit
```

**这不会让生成的 kernel 获得认证。** 该命令在 `EXPERIMENTAL` 策略下运行，打印
`UNCERTIFIED JIT OPT-IN` 警告，并记录所有获豁免的前提。应用程序代码也必须显式
传入 `policy="experimental", gpu_delivery="jit"`；这项风险确认不会持久化，也不会被
后续认证进程继承。使用结果前请检查
`explain()["experimental_waivers"]`。

经过修正并固定 Unicode 数据版本的 Gigatoken 实现已直接链接到核心 `toktier._native`
扩展中。TokTier 不会安装或信任名为 `gigatoken` 的顶层包，wheel 也不包含第二个
CPU 原生模块。基础 wheel 还固定了启用这条认证路径所需的 HF 加载器和参考实现
版本，无需另行安装 CPU-fast 组件。

如需核对来源，可在源码检出目录中独立重算当前源码身份，并使用相同的发布配置
构建：

```bash
python tools/fast_cpu_source_identity.py
maturin build --locked --release
```

[来源与构建记录](packaging/fast_cpu/README.md)中固定了上游 commit、补丁、Unicode
数据、编译器和发布参数。运行中的扩展会报告经过域分隔的源码摘要、精确 Rust
工具链和构建参数；注册表会在启用路由前，将这些信息与 repair 配置、参考实现和
tokenizer 工件一并验证。核心 wheel 还包含 Gigatoken 的 MIT 许可证、TokTier
修改声明、依赖 SBOM 和依赖许可证合集。

TokTier 当前只发布 ABI3 Linux x86-64 wheel，不发布 sdist。任意安装期重编译
都会产生不同的工具链/构建身份，因此在单独认证前会 fail-closed。带有 tag 的仓库
包含完整源码与固定构建记录；是否发布 sdist 仍是独立的发布决策。

预编译 fatbin 包含 `sm_75/80/86/89/90/100/120` 镜像以及
`compute_75` PTX fallback。绑定二进制摘要的认证覆盖 `sm_89` 和
`sm_120`；其他随包架构标记为 `experimental`。使用默认 facade 时，
`toktier[gpu]` 选择预编译交付，`toktier[gpu-jit]` 则根据检测到的配置选择
JIT；显式 `gpu_delivery=` 参数可覆盖该检测结果。预编译交付下，GPU 引擎会在
构造原生请求路径时初始化；无论首个请求大小如何，都会触发这一过程，因此
短请求过后 `explain()["gpu_backend"]["loaded"]` 也可能读到 `true`；crossover 仍会
逐个输入决定实际由哪个后端执行。JIT 交付继续使用 Python 主机端，其 GPU 后端直到
首个路由到 GPU 的输入到来时才会加载。JIT 交付在 `sm_89` 和
`sm_120` 上的状态为 `certified_source`，其证书绑定源码、类别表、编译参数
和工具链约束，而不是本机生成的二进制。自动 facade、显式引擎 API 和交付
诊断见 [`docs/gpu-jit.md`](docs/gpu-jit.md)。

tokenizer 工件不打包在 wheel 中，而是从固定的上游 revision 获取并用
SHA-256 验证。CLI 同时支持联网、镜像和气隙环境：

```bash
toktier artifacts fetch qwen3_8b
toktier artifacts export qwen3_8b --out qwen3_8b.tar
toktier artifacts import qwen3_8b.tar
toktier artifacts verify qwen3_8b
toktier inspect qwen3_8b
toktier doctor --json
```

这套流程搬运的只是 tokenizer 工件。真正断网的主机还需要另行准备 TokTier
wheel 及其全部依赖 wheel（wheelhouse 或本地索引）；bundle 格式本身不携带
任何 Python 发行包。

核心包不依赖 `torch`，导入时不需要 CUDA、网络连接或硬件探测。工件缓存、
编译 kernel 缓存和持久化会话状态分别存放：两个缓存跟随 `XDG_CACHE_HOME`，
会话 store 跟随 `XDG_STATE_HOME`——因为状态不属于缓存。要一次性迁移全部目录，
可使用 `TOKTIER_HOME`（见 `docs/contracts/config.md` 第 5 节）。

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

| 验证活动 | 规模 | 观测到的不一致 |
|---|---:|---:|
| 全语料差分验证 | 14 个工件 × 3,800,016,491 篇文档 = **53,200,230,874 次检查** | 0 |
| 语料体量 | 12,328,592,579,973 个 Unicode 码点 | — |
| 发布代码一致性验证 | 15,960,166 篇文档 | 0 |
| 修正版 Gigatoken CPU repair | 11 个唯一工件 × 3,800,016,491 篇文档 = **41,800,181,401 次检查**（通过精确工件继承覆盖 12 个 family） | 0 |

机器可读记录位于
[`evidence/evidence_manifest.json`](evidence/evidence_manifest.json)、
[`evidence/evidence_manifest_added_families.json`](evidence/evidence_manifest_added_families.json)
和 [`tables/support_registry.json`](tables/support_registry.json)。随版本提供的
逐工件测量记录覆盖 49,920,199,013 次检查；早期归档阶段覆盖其余
3,280,031,861 次，两者相加得到上面的总数。通过历史公开 session API 进行的
一次聚焦的端到端复验记录在
[`readings/fast_cpu_focused_parity.json`](readings/fast_cpu_focused_parity.json)；
实际执行的单次调用 Rust 前端则在全部 11 个支持 CPU 快速路径的工件上另行检查，记录在
[`readings/fast_cpu_native_frontend_parity.json`](readings/fast_cpu_native_frontend_parity.json)。

三个状态把证据与运行时行为区分开：

| 状态 | 含义 |
|---|---|
| `certified` | 证据绑定精确工件与加速二进制；预编译 GPU 交付还绑定 Rust 主机端的源码摘要、精确 rustc 与发布构建信息。 |
| `certified_source` | 证据绑定源码、构建输入与工具链；集成 CPU 引擎和本机 GPU JIT 使用此状态。 |
| `reference-only` | 不采用加速路径，运行 HF `tokenizers`。 |

这些是经验性差分结果，并非对所有可能输入的证明。针对每个请求的检查和参考
fallback 仍是系统契约的一部分。

仓库自检命令：

```bash
pip install pytest==9.1.1 jsonschema==4.26.0    # 或：pip install --group test
python tools/generate_evidence.py --check
python tools/generate_native_legal.py --check    # 需要 cargo
python tools/validate_registry.py tables/support_registry.json
python tools/generate_registry.py --release-check
python tools/generate_sibling_aliases.py --check
python tools/dev.py test-packaging
```

这里写明前置条件，是为了让一次失败确实代表问题本身，而不是缺少工具。
schema 检查需要 `jsonschema`（`pyproject.toml` 的 `test` 依赖组）；缺少时
它会明确拒绝并提示 `error: the jsonschema package is required ...`。最后一条
命令运行打包测试套件，还需要 `pytest`，同在该依赖组内——上面第一行直接
安装这两个固定版本，可满足命令块内所有命令的依赖。备选的 `pip install --group` 从
`pyproject.toml` 读取该组，需要 pip 25.1 或更新版本（PEP 735）。

`generate_native_legal.py` 读取工作区的锁定依赖解析图，需要 `cargo`
（由 `rust-toolchain.toml` 固定的工具链）。新克隆的仓库需先执行一次
`cargo fetch --locked` 填充本地 Cargo 缓存（需要网络；仅 `cargo build` 还不够，
因为法律合规检查覆盖所有 target），之后该检查本身可离线运行。
`generate_registry.py` 从已编译扩展中读取 native host 的编译期身份：如果本仓库的
`src/toktier/_native` 已构建，就使用该扩展，否则使用已安装 `toktier` wheel 中的
扩展，并在 stderr 说明读取了哪一个；无论哪种情况，该身份都必须与当前源码集合
完全一致。`pytest tests/gpu` 遵循同一规则：断言该身份的两项测试会读取可用的
扩展，只有两者都不可用时才会注明原因并跳过。

## 性能

README 顶部的图比较了对同一文本执行自动 GPU/repair 路由与完整重编码的结果。
发布的批量 GPU 路径还有一组同机吞吐数据：

| 路径 | 吞吐 | 环境 |
|---|---:|---|
| GPU 端到端（文本输入、ID 输出） | 0.6028 GB/s | 单张 RTX PRO 6000 Blackwell |
| HF 参考 CPU 路径 | 0.0047 GB/s | 同一主机、同一输入、单 CPU 核 |

这组数据使用 2.2 GB 驻留内存的真实网页文本，报告按墙钟时间计算的 UTF-8 字节
吞吐量，并物化主机侧 ID 数组。完整协议、每个数据单元格和溯源信息见
[`docs/benchmarks.md`](docs/benchmarks.md)。

主要实验使用 RTX PRO 6000 Blackwell，但消费级 RTX 5090 在相同协议下的一轮
测试反而**快 11–17%**（所报告 family 的吞吐量为 4.24–5.50 GB/s）。因此，
消费级硬件是实际可行的部署目标，并非降级模式：RTX 4090 也通过了针对 `sm_89`
的整套正确性与预编译交付测试。这些观察结果并不保证每张 GPU 都有相同比例的
提升；实际性能仍取决于架构、负载和主机端交付方式。

![单请求延迟](docs/figures/f1_single_request_latency.svg)

![会话尾延迟](docs/figures/f2_session_tail_latency.svg)

![会话状态内存](docs/figures/f3_session_state_memory.svg)

![repair 路径的等效吞吐](docs/figures/f4_repair_equivalent_throughput.svg)

图中明确标注了 Hugging Face（HF）`tokenizers`，并注明每幅图对应的
`docs/figures/*.data.json` 机器可读文件。基准文档还展示了直接使用其他引擎
速度更快的适用区间。

## 支持矩阵

| 类别 | family 数量 | 覆盖情况 |
|---|---:|---|
| 认证 CPU 快速 repair | 12 个 family / 11 个唯一 tokenizer 工件 | 修正版 Gigatoken，12.33 万亿字符，未观察到 ID 不一致 |
| Byte-level BPE | 15 | CPU 证据；逐工件记录 GPU 状态 |
| WordPiece | 3 | CPU 证据 |
| 结构性排除 | 2 | 记录具体原因 |

[`docs/support-matrix.md`](docs/support-matrix.md) 列出了每个锚点工件、
SHA-256、后端状态，以及 **210 个已验证模型仓库**；这些仓库使用完全相同或
仅序列化形式不同的 tokenizer。覆盖关系由 tokenizer 内容决定，而不是仓库名。
`toktier.from_pretrained(repo_id)` 会在运行时落实这条规则：对解析到的文件
计算哈希，将登记内容映射到 canonical 工件，其他内容继续使用 HF。

210 个 sibling 条目中，191 个会映射到当前 wheel 随附的 canonical 工件。
其余 19 个不会获得加速准入，因为对应 canonical 工件尚未打包：7 个
WordPiece 条目使用 HF，12 个源码级 `kimi_k3` 条目目前还需要转换工件，因而
会给出可操作的错误，而不会假装可以直接加载 `tiktoken.model`。
`toktier inspect` 仍是随包 family 列表的权威来源。

## 与现有工作的关系

Incremental BPE 研究关注 merge 阶段如何随新字节到来而扩展。toktier 工作在它的
上一层：会话状态保存整个 tokenizer 处理流程的 token ID 和 span，涵盖
normalization、pre-tokenization、merge 和 added-token 处理；只有边界检查通过
后才会接受追加内容。

`llm-tokenizer` 和 NVIDIA Dynamo 的 `dynamo-tokenizers` 等服务项目也会
缓存编码结果。主要接口差异如下：

| 属性 | toktier | 进程内前缀缓存 |
|---|---|---|
| 生命周期 | 可持久化、跨进程 | 跟随 tokenizer 进程 |
| 命中验证 | 摘要用于筛选候选，再由已存字节验证 | 以摘要为键进行 lookup |
| 复用边界 | 经过认证的 tokenizer 边界 | 通常是 special-token 边界 |
| 使用界面 | 供自行管理会话状态的应用使用的 Python 库 | 服务网关组件 |

两层共同使用的方法见
[`docs/integration/dynamo.md`](docs/integration/dynamo.md)。

## 文档

- [`docs/releases/v0.2.0.md`](docs/releases/v0.2.0.md) — 本版本发布说明（英文）。
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 分层、路由与 store 格式。
- [`ROADMAP.md`](ROADMAP.md) — 发布范围与后续集成。
- [`docs/support-matrix.md`](docs/support-matrix.md) — 工件与覆盖仓库。
- [`docs/gpu-jit.md`](docs/gpu-jit.md) — 预编译和 JIT GPU 交付。
- [`docs/rust-api.md`](docs/rust-api.md) — 无需 Python 的 Rust serving API。
- [`docs/rust-lifecycle.md`](docs/rust-lifecycle.md) — 原生工件、direct JIT、并发与离线分发。
- [`docs/integration/dynamo.md`](docs/integration/dynamo.md) — Dynamo 集成。
- [`docs/paper/toktier-preprint.pdf`](docs/paper/toktier-preprint.pdf) — 当前论文预印本。

## 致谢

TokTier 的 CPU Fast Pass 和 Fast Repair 建立在
[Gigatoken](https://github.com/marcelroed/gigatoken) 与
[Fastokens](https://github.com/crusoecloud/fastokens) 两项优秀的开源工作之上。感谢
两项工作的作者和贡献者将这些成果开源。

修正版 Gigatoken 是 11 个唯一 tokenizer 工件的默认认证 repair-window 引擎；
由于 NVIDIA Nemotron-Terminal 随附了逐字节完全相同的 `qwen3_8b` tokenizer，
这些工件覆盖 12 个 family。TokTier 的兼容性补丁使 Gigatoken 的 Unicode
数据和 UTF-8 处理与冻结的
[Hugging Face tokenizers](https://github.com/huggingface/tokenizers) 参考实现对齐；
该路径在 12.33 万亿字符上完成 418 亿次检查，未观察到 token ID 不一致。

Fastokens 0.3.1 保留为显式实验性备选；TokTier 不声称该适配器的 ID 与参考实现
完全一致，也不会自动选择它。Fastokens 采用 Apache-2.0，Gigatoken 采用 MIT；
确切 revision、许可证副本、Gigatoken 补丁与修改声明见
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES) 和 [`packaging/`](packaging/)。

## 许可证与引用

toktier 采用 [Apache License 2.0](LICENSE)；署名信息见
[`NOTICE`](NOTICE)。

**论文：** [*TokTier: Exact Stateful CPU+GPU Tokenization for Agentic LLM
Serving*](https://arxiv.org/abs/2607.29678) ·
[PDF](https://arxiv.org/pdf/2607.29678)

如果你在研究中使用 toktier，请引用：

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
