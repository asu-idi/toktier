# toktier

[English](README.md) | **简体中文**

**整段对话只分词一次，之后只处理新增内容。**

TokTier 是面向智能体 LLM 服务的有状态分词系统。它保留每个会话的 token
状态，通过认证 CPU 路径对追加文本执行 repair，并为全新请求或大型请求提供
认证 GPU 路径。两条快速路径返回的 token ID 序列，都与 Hugging Face（HF）
`tokenizers` 从头完整编码得到的 token ID 序列**完全一致（bit-identical）**。

- **大规模精确验证。** 发布验证覆盖 15 个 tokenizer 工件、38 亿篇真实文档，
  共记录 **570 亿次检查**（12.33 万亿字符），未观察到任何不一致。
- **CPU 与 GPU 均有快速路径。** 在已记录的整套基准测试中，GPU 路径处理一个
  全新的 400 万字符请求（约 78.6 万 token）用时 **3.88 ms**；对 419 万字符
  会话追加 256 个字符时，有界 CPU repair 为 **1.68 ms**。
  [基准协议](docs/benchmarks.md)明确排除引擎构造开销，因此 **3.88 ms**
  的前提是引擎已构造并就绪，不是冷启动首次调用的数字。repair 读数只测量
  repair 操作本身，
  不包含将完整历史 token 序列物化为 Python tuple 的成本。
- **先认证，再加速。** 只有精确的 tokenizer 工件、参考实现版本、kernel 交付方式
  和架构均有记录证据覆盖时，系统才会采用快速路径。`explain()` 会报告实际路由
  及其原因。

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="docs/figures/hero_session_vs_reencode_dark.svg">
  <img alt="toktier 与完整重编码在三种 400 万字符量级负载下的延迟对比，线性坐标"
       src="docs/figures/hero_session_vs_reencode.svg">
</picture>

图中每根柱均表示实测中位数。上文的 1.68 ms 来自
`toktier repair (HF tokenizers window)` 测试系列，也就是取得该读数时采用的
repair 窗口；修正版 Gigatoken 窗口则属于同一图中的另一个测试系列（追加
65,536 个字符时为 2.39 ms）。两者都是下方路由表涵盖的有界 repair，图表
数据标明了每根柱所属的测试系列。原生 Rust serving 集成可通过保留会话状态
并仅使用 repair 后的 token 后缀，避免物化完整历史序列。精确数值、负载大小和
样本数见
[`hero_session_vs_reencode.data.json`](docs/figures/hero_session_vs_reencode.data.json)，
完整的扫描测试结果见 [`docs/benchmarks.md`](docs/benchmarks.md)。

## 最新动态

- **2026.08.TBD** 🚀 **toktier 0.2.6** 发布——Rust 侧认证现在只为
  certified core 说话（TokTier 自家的 crate、它们直接调用的包，以及
  它们下面真正参与分词计算的文本语义库）；编译闭包其余部分
  的版本差异改以 advisory 形式如实报告并附对齐命令，不再拦下加速
  路径；整闭包的严格读数仍保留在 `dependency_closure`。决定文本切点的
  那几个库，改按它们携带的 Unicode 表的版本比对，而不是按 crate
  版本；`doctor` 逐一报出。快速 CPU 预分词器读的属性数据已钉到参考
  引擎所携带的同一个 Unicode 版本，并有穷举对拍的门看住两边一致。
  GPU 这一侧：驱动与 CUDA 版本改为按"环境事实"如实报告，不再作为证书前提；
  认证活动没有测过的设备架构或编译工具链，在新的默认 `supported` 策略下会
  照常运行并标注为 `supported_untested`；`toktier-rust verify-local` /
  `toktier verify-local` 可以用你自己的文本把这条路与参考引擎逐 id 对一遍，
  通过则记为 `locally_verified`（这是一次本机测量的记录，不是证书）。
  对外返回的 ID、store 格式与 kernel ABI 都没有变化。详见
  [v0.2.6 发布说明](docs/releases/v0.2.6.md)（英文）。
- **2026.08.15** 🚀 **toktier 0.2.5** 发布——Rust crate 的 `network` feature
  改为按需开启，这让可能从不联网取件的默认构建少带 16 个包和整套 TLS 栈；
  仍需通过网络获取工件时加上 `features = ["network"]`，命令行则用
  `cargo install --locked --features network toktier`。Python 包的取件方式
  与此前完全相同，这项 feature 变化够不到它；真正会被 Python 侧看到的是
  `Session.revision` 自本版起可跨进程持久：在后续进程中恢复的会话报出其
  记录携带的 revision，`Tokenizer.store_session_revision()` 也由 `None`
  改为报出该数值。诊断面新增执行 `reason` 与 `network_compiled` 构建事实。
  对外返回的 ID、store 格式与 kernel ABI 都没有变化。详见
  [v0.2.5 发布说明](docs/releases/v0.2.5.md)（英文）。
- **2026.08.14** 🚀 **toktier 0.2.4** 发布——Han family（`kimi_k3`）以产品
  自带的端到端 GPU 引擎正式进入认证名单；Rust 侧认证开始判定"这次构建真正
  编译的那些包"，而不只是源码，build flags 也只声称 build script 观测得到的
  项；Rust crate 开始遵循 `TOKTIER_HOME`/XDG；Python facade 新增 `session()`
  上下文管理器、全命令 `--json` 与 `doctor --family`；无法承载私有状态的
  目录根、`artifacts check-conversion`、以及 last-execution 诊断，都在各自的
  契约内作答。对外返回的 ID、store 格式与 kernel ABI 都没有变化。详见
  [v0.2.4 发布说明](docs/releases/v0.2.4.md)（英文）。
- **2026.08.11** 🚀 **toktier 0.2.1** 发布——维护版本：诊断信息更完整
  （`doctor` 会报告 JIT 工具链是否合格，`explain()` 摘要逐项标明所指的时间
  范围），并修正了文档；对外返回的 ID、store 格式与 kernel ABI 都没有变化。
  详见 [v0.2.1 发布说明](docs/releases/v0.2.1.md)。
- **2026.08.10** 🚀 **toktier 0.2.0** 发布——首个公开版本：提供采用有界
  CPU repair、ID 精确性已获认证的会话，以及预编译 GPU 路径和 Rust
  serving API；分别以 Python wheel（[PyPI](https://pypi.org/project/toktier/)）
  和六个 Rust crate（[crates.io](https://crates.io/crates/toktier)）的
  形式发行。
- **2026.07.31** 📄 论文 [*TokTier: Exact Stateful CPU+GPU Tokenization
  for Agentic LLM Serving*](https://arxiv.org/abs/2607.29678) 上线 arXiv。

## 快速开始

运行 `pip install toktier` 即可安装；GPU 选项见[安装](#安装)。

```python
import toktier

tok = toktier.load("qwen3_8b")          # 支持矩阵中的 family id
enc = tok.encode("hello world")         # token ID
print(enc.ids)
print(tok.decode(enc.ids))
print(tok.explain(summary=True))        # 简明路由与判定
```

先 `encode` 再 `decode`，不一定能还原为完全相同的文本：如果 tokenizer 的处理
流程包含归一化（例如 NFC），返回的会是归一化后的文本。这是 tokenizer 自身的
行为，并非 TokTier 造成的不一致。TokTier 给出的保证是关于 ID 的，这项保证不受影响：
对于同一输入，其 ID 与 HF 从头编码的结果相同，而且两个解码器返回的文本也相同。

如果应用代码拿到的是 Hugging Face 模型仓库，而不是 TokTier family id，也可以
按 tokenizer 内容解析：

```python
tok = toktier.from_pretrained("Qwen/Qwen3-0.6B")
```

对于已登记的 sibling 或 canonical 仓库，`from_pretrained()` 会下载经审计的
不可变 revision，对解析到的那个文件计算 SHA-256，再查询由根摘要校验的
sibling 注册表（210 个已审计仓库，另加 1 个 canonical 自指条目）。对于未登记的仓库，`from_pretrained()`
在未显式传入 `revision=` 时解析 `main`。字节完全相同、经 canonicalization
后等价或序列化等价的记录，会通过同一套 CPU/GPU 路由使用已经认证的
canonical 工件。已知仓库的字节内容一旦变化，或遇到任何未登记
内容，在允许参考实现 fallback 的策略下都会继续使用 HF；
`REQUIRE_ACCELERATED` 策略则会报错。`explain()["model_resolution"]` 会同时报告
来源身份和实际执行的 canonical 身份。`load(family)` 仍是直接的 family API，
也是适合气隙环境的路径。

### 会话

为不断增长的对话记录指定 `session=`，即可跨调用和进程保存 token 状态：

```python
tok = toktier.load("qwen3_8b", store="./toktier-store")

transcript = "user: hello\nassistant: hello! how can I help?\n"
enc = tok.encode(transcript, session="chat-42")

transcript += "user: what changed since my last call?\n"
enc = tok.encode(transcript, session="chat-42")

# 使用 store 的路径与从头编码路径返回相同的 ID。
assert enc.ids == tok.encode(transcript, lookup="off").ids
```

不传 `session=` 时，store 也可以按内容查找经过逐字节验证的已存前缀。传入
`lookup="off"` 可跳过此次查找。逐字节验证失败只会被视为未命中，绝不会当作
可信命中；缓存淘汰只影响延迟，不影响输出。封存长会话的稳定前缀后，该前缀
在重启后仍可复用：TokTier 会先将记录与调用方提供的历史前缀绑定，再恢复
记录；绑定缺失或损坏时则执行冷编码。

### 路由与策略

路由策略可以选择，也可以查看：

```python
from toktier import RoutingPolicy

tok = toktier.load("qwen3_8b", policy=RoutingPolicy.CERTIFIED)
```

| 策略 | 执行的路由 | 快速路径前提不成立时 |
|---|---|---|
| `SUPPORTED`（0.2.6 起为默认） | `CERTIFIED` 放行的全部路由，另外还包括认证活动没有测过的设备架构或编译工具链——前提是随包 kernel 在那里能装载并运行，且注册表确实绑定的每一项约束都仍然通过；这样的路由标为 `supported_untested`，而不是 certified | fallback 到 HF，并记录原因 |
| `CERTIFIED`（严格档；0.2.5 及以前的默认） | 只执行精确工件、HF 版本、引擎/kernel 字节、交付方式和硬件均有证据覆盖的路由 | fallback 到 HF，并记录原因 |
| `REFERENCE` | 只使用 HF `tokenizers` | 不尝试任何加速路径 |
| `REQUIRE_ACCELERATED` | 与默认策略相同的路由 | 若构造时没有合格快速路径则报错；仍启用针对单个输入的安全 fallback |
| `EXPERIMENTAL` | 评估时可以采用未经评审的组合 | 明确标出每项获豁免的前提；永远不是默认策略 |

在此基础上，安装配置和输入形态会决定自动路由：

| 默认 `SUPPORTED` 策略下的情形 | 自动路由 |
|---|---|
| 安装项为 `toktier`，且使用 11 个认证 tokenizer 工件（覆盖 12 个 family）之一 | 修正版 Gigatoken 完成 CPU 全量编码；任一绑定检查失败则使用 HF |
| 安装项为 `toktier[gpu]`，且冷请求/普通请求的输入大小低于 GPU crossover（默认 64 KiB） | 修正版 Gigatoken CPU 路径（没有 CPU-fast 认证的 family 使用 HF） |
| 安装项为 `toktier[gpu]`，且冷请求/普通请求的输入大小达到或超过 GPU crossover（默认 64 KiB） | 随包预编译 GPU 路径；随后按固定 fallback 链依次使用修正版 Gigatoken 和 HF |
| 已存在的会话收到严格追加 | 覆盖范围内的 12 个 family 使用修正版 Gigatoken CPU repair，不受完整对话总长度影响 |
| Added-token 或 repair guard 无法证明前提 | 该输入使用 HF 参考路径 |

表中两行 GPU 描述的是 GPU 路由被准入之后的行为，而准入条件比"机器上有 GPU"
更窄。随包证据覆盖到的架构为：预编译交付的 `sm_80`、`sm_89`、`sm_90` 与
`sm_120`。在其他架构上，默认的 `SUPPORTED` 策略仍会运行随包 kernel——只要它
能装载、且注册表确实绑定的每一项约束都通过——并把该路由标为
`supported_untested` 而不是 certified；`toktier verify-local` 可以用你自己的
文本把它与参考引擎对一遍。严格的 `CERTIFIED` 策略则会拒绝它，这两行随之落到
上一行。两种情况下 `explain()` 都会记录原因。逐架构的证据规模见
[`docs/support-matrix.md`](docs/support-matrix.md#status-vocabulary)。

`explain(summary=True)` 报告：

- 主要路由与认证结论；
- 最近一次请求的实际执行情况（`last_execution_backend` / `_path` /
  `_source`，如果请求结束时的后端与起始后端不同，还会给出
  `last_execution_fallback`）；
- 整个进程生命周期内是否发生过 fallback（`fallback_ever_occurred`，
  常规的低于阈值 crossover 路由也计入其中）。

完整的无参数 `explain()` 报告则再给出固定路由链、crossover 判定
（`gpu_min_bytes`，默认 64 KiB）、详细的探测与认证数据，以及每类
fallback 的计数。

## Rust serving API

工作区（Cargo workspace）提供无需 Python 的 Rust serving facade，供直接
保留 token 状态的前端使用。它提供：

- 固定版本工件的获取、镜像与气隙操作
- 参考实现、修正版 CPU、预编译或 direct-JIT 的 GPU 路由
- 连续 token 缓冲区
- 不依赖 executor 的有界批处理
- 持久化的命名会话
- 原生增量的 `TokenPatch` 结果

```rust
use toktier::{Device, Runtime};

let runtime = Runtime::builder().device(Device::Auto).build()?;
let tokenizer = runtime.load("qwen3_8b")?;
let mut session = tokenizer.open_session("agent-42")?;
let seed = session.seed("user: hello\n")?;
let patch = session.append("assistant: hi\n")?;
```

`patch.keep_tokens()` 指出下游保留的 ID 缓冲区应当截断到哪里；
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
cargo add toktier --features network # 同上，并加上通过 TLS 获取工件的能力
```

Python 包不受 Rust crate feature 的影响：它一如既往通过 `huggingface-hub`
获取工件。Rust 侧自 0.2.5 起 `network` 改为按需开启；不开启时，crate 仍可
校验、镜像、导入导出工件，并基于已校验缓存运行。

| 安装项 | 交付内容 | 要求 |
|---|---|---|
| `toktier` | 修正版 Gigatoken CPU 全量编码与会话 repair、HF fallback、持久化 store、路由和 CLI | Linux x86_64、glibc 2.34+、CPython 3.10+；固定安装 `tokenizers==0.22.2` 与 `transformers==4.57.6` |
| `toktier[gpu]` | `toktier` 的严格超集；通过 64 KiB crossover 自动路由到随包的多架构 CUDA fatbin | NVIDIA GPU、驱动版本 580.65.06+、`torch`；无需编译器，首次使用不编译 |
| `toktier[gpu-jit]` | CPU/GPU 路由与 `toktier[gpu]` 相同；在本机编译认证 kernel 源码 | 经评审的 NVCC / torch-runtime CUDA / PyTorch 三元组、`torch`、`ninja`；首次使用需要编译 |

两个 GPU extras 都会引入 `torch` 及其 CUDA wheel，请预留空间：全新的
`[gpu]` 或 `[gpu-jit]` 虚拟环境实测约 5 GiB，无缓存安装会下载数个几百 MB 级
的 wheel（pip 缓存规模相当）。这是 Torch 生态的体量，而非 TokTier 自身——基础
`toktier` wheel 完全不需要这些。

### JIT 工具链认证

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
`UNCERTIFIED JIT OPT-IN` 警告，并记录所有获豁免的前提。应用代码也必须显式
传入 `policy="experimental", gpu_delivery="jit"`；这项风险确认刻意不做持久化，
也不会被后续认证进程继承。使用结果前请检查
`explain()["experimental_waivers"]`。

### CPU 引擎来源与构建身份

经过修正并固定 Unicode 数据版本的 Gigatoken 实现已直接链接到核心 `toktier._native`
扩展中。TokTier 不会安装或信任名为 `gigatoken` 的顶层包，wheel 也不包含第二个
CPU 原生模块。基础 wheel 还固定了启用这条认证路径所需的 HF 加载器和参考实现
版本，无需另行安装 CPU-fast 组件。

如需核对来源，可在源码检出目录中独立重算当前源码身份，并使用相同的发布配置
构建：

```bash
python tools/fast_cpu_source_identity.py
python tools/compute_identity_v2.py
python tools/compute_identity_v2.py --show-diff
maturin build --locked --release
```

三个既有身份脚本（`fast_cpu_source_identity.py`、
`native_host_source_identity.py`、`rust_api_source_identity.py`）仍提供
当前构建信息所使用的逐字节 v1 视图。`compute_identity_v2.py` 仅归一化明确
列出的工作区版本字段，随后在各自的新域中对同一批 fast-CPU、native-host 与
Rust-API 覆盖集计算哈希；`--show-diff` 会逐行打印归一化后的内容，便于
审阅。`tools/dev.py check` 还会拒绝受覆盖的 Rust 或 Python 代码在明确列出
的构建信息报告位置之外读取包版本，从而确保允许的元数据变化不会影响运行时
行为。

[来源与构建记录](packaging/fast_cpu/README.md)中固定了上游 commit、补丁、Unicode
数据、编译器和发布参数。运行中的扩展会报告经过域分隔的源码摘要、精确 Rust
工具链和构建参数；注册表会在启用路由前，将这些信息与 repair 配置、参考实现和
tokenizer 工件一并验证。核心 wheel 还包含 Gigatoken 的 MIT 许可证、TokTier
修改声明、依赖 SBOM 和依赖许可证合集。

TokTier 当前只发布 ABI3 Linux x86-64 wheel，不发布 sdist。任意安装期重编译
都会产生不同的工具链/构建身份，因此在单独认证前会 fail-closed。带有 tag 的仓库
包含完整源码与固定构建记录；是否发布 sdist 仍是独立的发布决策。

### GPU 交付

预编译 fatbin 包含 `sm_75/80/86/89/90/100/120` 镜像以及
`compute_75` PTX fallback。绑定二进制摘要的认证覆盖
`sm_80`、`sm_89`、`sm_90` 与 `sm_120`；其他随包架构标记为 `experimental`。使用默认 facade 时，
`toktier[gpu]` 选择预编译交付，`toktier[gpu-jit]` 则根据检测到的配置选择
JIT；显式 `gpu_delivery=` 参数可覆盖该检测结果。预编译交付下，GPU 引擎按需
初始化：只有首个路由到 GPU（达到或超过 crossover）的请求才会触发加载，因此
在只跑过短请求的情况下，`explain()["gpu_backend"]["loaded"]` 会保持 `false`；crossover 仍会
逐个输入决定实际由哪个后端执行。JIT 交付继续使用 Python 主机端，其 GPU 后端
同样直到首个路由到 GPU 的输入到来时才会加载。JIT 交付在 `sm_89` 和
`sm_120` 上的状态为 `certified_source`，其证书绑定源码、类别表、编译参数
和工具链约束，而不是本机生成的二进制。自动 facade、显式引擎 API 和交付
诊断见 [`docs/gpu-jit.md`](docs/gpu-jit.md)。

### tokenizer 工件、镜像与气隙主机

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

### doctor：这台机器实际会走哪条路

`toktier doctor` 只做探测，从不加载 CUDA kernel。`devices` 中的每个条目
都会报告序号、名称和架构；`driver_version` 报告共享主机探针检测到的驱动
版本；`automatic_gpu_delivery_certification` 将检测到的每种架构映射到
所选安装配置对应交付方式的认证状态。`cuda_available` 报告 TokTier 的
CUDA 运行时绑定是否已安装；`cuda_hardware_present` 报告设备探测是否至少
找到一台可用的 CUDA 设备。

它还会回答“这台机器实际会走哪条路”，且不构造 tokenizer、不尝试编译：

| 字段 | 含义 |
|---|---|
| `automatic_gpu_candidate` | 仅反映安装层面：`torch` 可导入，且配置未禁用 GPU；并非合格性判定 |
| `jit_toolchain_satisfied` | JIT 交付下，本机检测到的编译器/运行时三元组是否属于注册表已评审的组合；预编译交付没有这项前提，因此报告 `null` |
| `jit_toolchain_observed` / `jit_toolchain_constraint` | 本机检测到的三元组，以及用于对照的已评审集合 |
| `automatic_gpu_eligible` | 以下条件的合取结果：候选条件成立；至少检测到一台设备，且其架构已通过所选交付方式的评审；该交付方式自身的材料齐备；工具链前提成立 |
| `automatic_effective_backend` | 对具备 CPU 快路径认证的 family，达到或超过 crossover 的自动请求实际使用的后端：`gpu`、`fast_cpu` 或 `hf` |
| `directory_roots_usable` / `directory_roots_problem` | 上述三个已解析目录根能否承载其用途，以及无法承载时的原因——与下一条命令会以 `CONFIG_INVALID` 作答的判断相同，且只读取、不创建任何目录 |

上述字段描述的是这套安装。`toktier doctor --family FAMILY` 会另加一个
`family` 段，回答某一个 family 在这台机器上的情况：它的认证身份、
`fast_cpu` 与 GPU 状态，以及两个实际后端——一个对应达到或超过 crossover
的请求，一个对应低于 crossover 的请求。差别正体现在后者：CPU 车道走参考
实现的 family 在低于 crossover 时读作 `hf`，而安装层面的字段（同样属实）
读作 `fast_cpu`。

因此，在未经评审的编译器上安装 `toktier[gpu-jit]`，会同时报告
`automatic_gpu_candidate: true`、`jit_toolchain_satisfied: false`、
`automatic_gpu_eligible: false` 与 `automatic_effective_backend: fast_cpu`
——其结论与 `toktier gpu compile` 一致，而且无需先运行该命令。

### 缓存、状态与目录布局

核心包不依赖 `torch`，导入时无需 CUDA、网络连接或硬件探测。工件缓存、
已编译 kernel 缓存和持久化会话状态分别使用独立目录：前两类缓存遵循
`XDG_CACHE_HOME`，会话 store 遵循 `XDG_STATE_HOME`——因为状态不属于缓存。
如需统一指定所有目录的位置，可使用 `TOKTIER_HOME`（见
`docs/contracts/config.md` 第 5 节）。

自 0.2.4 起，Rust crate 以相同的优先级读取同一组变量，一套环境即可同时
安置两层：

| 层 | 工件缓存 | 已编译 kernel 缓存 | 会话状态 |
|---|---|---|---|
| Python（`toktier`） | `TOKTIER_HOME` / `XDG_CACHE_HOME` | `TOKTIER_HOME` / `XDG_CACHE_HOME` | `TOKTIER_HOME` / `XDG_STATE_HOME` |
| Rust（`toktier` crate） | `TOKTIER_ARTIFACT_CACHE`，否则同一组根目录 | `TOKTIER_JIT_CACHE`，否则同一组根目录（`jit` feature） | `RuntimeBuilder::home()`，否则 `TOKTIER_HOME` / `XDG_STATE_HOME` |

末级目录名仍是 crate 自己的：工件与 Python 产品同名并列，JIT 产物放在
`jit-rust`，与 Python 的 `kernels` 目录有意区分，因为两者存放的东西不同。
若三个根目录都没有设置，持久化会话会直接报错而不是凭猜测选一个位置——
状态不是缓存。参见 [`docs/rust-lifecycle.md`](docs/rust-lifecycle.md)。

### 实验性：Fastokens 对照

Fastokens 0.3.1 只作为显式实验性对照使用：

```python
tok = toktier.load(
    "qwen3_8b", policy="experimental", repair_backend="fastokens"
)
```

该适配器会重新编码完整会话，并报告 `exact_id_guarantee: false`；认证策略永远
不会自动选择它，它也不在修正版 Gigatoken 的 12.4 TB（12.33 万亿字符）声明
的覆盖范围内。

这一判断现在有实测支撑。在以 `tokenizers==0.22.2` 为参照、规模逐步放大、
最高到每个工件 998,857,881 篇的一系列差分中，我们观察到上游 0.3.1
代码相对该参照存在五处缺陷：一处会在罕见字符上报错，另外四处是静默的 id
不一致——不抛异常、也没有任何提示，调用方无法从返回值看出差别。五处各有
一个独立复现件，在由上游原始源码构建出的版本上都会失败。相关报告已为上游
项目准备好。在这些源码上应用我们自己的五个补丁后，最终一轮差分——每个工件
998,857,881 篇、共 15 个 tokenizer 工件——录得零处 id 不一致，口径是一个
108 码位的守卫，它会把一小部分文档改走参考实现路径。

上述读数描述的是**源码修订版本**，不是某个二进制。`explain()` 报告的
`engine_digest` 同时覆盖编译出的扩展模块，因此由同一份源码在另一台机器上
构建的轮子会报告不同的摘要——这些数字背后的两只补丁轮子分别报告
`4c7e9c9f03313cc0...` 与 `24e3cccaef6d97f4...`，作为对照的上游原始源码构建
报告 `8600f509c10dc03c...`。所有构建的版本号都是 `0.3.1`，因此版本号区分不
了它们；完整摘要见 `docs/support-matrix.md`。若你的安装报告的是别的摘要，
上述数字就不是在描述它。该适配器的状态没有变化：仍是显式实验性、仍需显式
选择、仍不会被自动选中。

## 正确性与证据

认证参考实现是默认配置下的 Hugging Face `tokenizers` 0.22.2。认证绑定
精确的工件字节和参考版本。如果本机 HF 版本不在认证集合内，加速路由会被
关闭，请求继续使用本机安装的参考路径。

本文档中出现了四类计数，各自回答不同的问题：**15 个随包工件**（`toktier
inspect` 列出的那些）、**15+3 个 model family**（byte-level BPE 加
WordPiece；不同 family 可以共用同一工件）、**11 个具备 CPU 快路径认证的
工件**（按精确工件继承覆盖 12 个 family），以及**注册表 211 条**（210 个
已审计 sibling 仓库加 1 个 canonical 自指条目）。某个数字看起来与另一个
对不上时，通常是因为它们属于不同的计数轴。

| 验证活动 | 规模 | 观测到的不一致 |
|---|---:|---:|
| 全语料差分验证 | 15 个工件 × 3,800,016,491 篇文档 = **57,000,247,365 次检查** | 0 |
| 语料体量 | 12,328,592,579,973 个 Unicode 码点 | — |
| 发布代码一致性验证 | 15,960,166 篇文档 | 0 |
| 修正版 Gigatoken CPU repair | 11 个唯一工件 × 3,800,016,491 篇文档 = **41,800,181,401 次检查**（通过精确工件继承覆盖 12 个 family） | 0 |

机器可读记录位于
[`evidence/evidence_manifest.json`](evidence/evidence_manifest.json)、
[`evidence/evidence_manifest_added_families.json`](evidence/evidence_manifest_added_families.json)、
[`evidence/evidence_manifest_kimi_band.json`](evidence/evidence_manifest_kimi_band.json)
和 [`tables/support_registry.json`](tables/support_registry.json)。随版本提供的
逐工件测量记录覆盖 53,720,215,504 次检查；早期归档阶段覆盖其余
3,280,031,861 次，两者相加得到上面的总数。经由既有的公开会话 API 做的
一次针对性端到端复验，记录在
[`readings/fast_cpu_focused_parity.json`](readings/fast_cpu_focused_parity.json)；
实际执行的单次调用 Rust 前端则在全部 11 个支持 CPU 快速路径的工件上另行
检查，记录在
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
python tools/verify_carryover.py --check
python tools/generate_native_legal.py --check    # 需要 cargo
python tools/validate_registry.py tables/support_registry.json
python tools/generate_registry.py --release-check
python tools/generate_sibling_aliases.py --check
python tools/dev.py test-packaging
```

其中五条同样可以在发布的 Rust 源码归档里跑通——该归档随源码带上了
`evidence/`、`data/` 以及本 README 的译本。另外两条只在仓库内可用；在归档里
它们会明确声明不适用，而不是给出一次莫名的失败：各自打印一行以 `declined:`
开头的说明，指出未检查任何内容，并以 `3` 退出——既不是通过，也不是发现问题。
`generate_registry.py --release-check` 读取仓库自身的源码树与其已构建扩展，
而归档两者皆无；归档所携带的注册表副本由 `validate_registry.py` 验证，那一条
在归档里确实可跑。`dev.py test-packaging` 运行测试套件，而归档有意不携带它。
两棵树的区分依据是归档构建器写在其根目录（且仅写在那里）的
`SOURCE-MANIFEST.json`，因此仓库内的检出仍与此前完全一样地跑满七条。

| 退出码 | 该工具在说什么 |
|---|---|
| `0` | 已运行，且所检查的内容成立 |
| `3` | 已声明不适用：这不是它能检查的树，未检查也未运行任何内容 |
| 其他 | 已运行并发现了问题，或无法运行——具体由消息说明 |

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
完全一致。第二种读取仅是仓库内的便利做法——在发布的源码归档里该命令会直接
声明不适用，而不是去回答一个属于本机上其他安装的扩展。`pytest tests/gpu` 遵循同一规则：断言该身份的两项测试会读取可用的
扩展，只有两者都不可用时才会注明原因并跳过。

## 性能

README 顶部的图比较了对同一文本执行自动 GPU/repair 路由与完整重编码的结果。
发布的批量 GPU 路径还有一组同机吞吐数据：

| 路径 | 吞吐 | 环境 |
|---|---:|---|
| GPU 端到端（文本输入、ID 输出） | 0.6028 GB/s | 单张 RTX PRO 6000 Blackwell |
| HF 参考 CPU 路径 | 0.0047 GB/s | 同一主机、同一输入、单 CPU 核 |

这组数据使用常驻内存中的 2.2 GB 真实网页文本，在物化主机侧 ID 数组的前提
下，报告按墙钟时间计算的 UTF-8 字节吞吐。完整协议、每个数据单元格和溯源
信息见 [`docs/benchmarks.md`](docs/benchmarks.md)。

主要实验使用 RTX PRO 6000 Blackwell，但消费级 RTX 5090 在相同协议下的一轮
测试反而**快 11–17%**（所报告 family 的吞吐量为 4.24–5.50 GB/s）。因此，
消费级硬件是实际可行的部署目标，并非降级模式：RTX 4090 也通过了针对 `sm_89`
的整套正确性与预编译交付测试。这些观察结果并不保证每张 GPU 都有相同比例的
提升；实际性能仍取决于架构、负载和主机端交付方式。

图中明确标注了 Hugging Face（HF）`tokenizers`，并注明每幅图对应的
`docs/figures/*.data.json` 机器可读文件。基准文档还展示了直接使用其他引擎
速度更快的适用区间。

![单请求延迟](docs/figures/f1_single_request_latency.svg)

![会话尾延迟](docs/figures/f2_session_tail_latency.svg)

![会话状态内存](docs/figures/f3_session_state_memory.svg)

![repair 路径的等效吞吐](docs/figures/f4_repair_equivalent_throughput.svg)

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

随包注册表共 211 条：上述 210 个 sibling，外加 `moonshotai/Kimi-K3`
自身——这样按名解析 canonical 仓库时，报告的 evidence 仓库就是它自己，
而不是某个字节完全相同的 sibling。其中 204 条会映射到当前 wheel 随附的
canonical 工件。其余 7 个是 WordPiece 条目，对应 canonical 工件尚未打包，
因而使用 HF。13 个源码级 `kimi_k3` 条目已在这 204 个之内：其 canonical
工件由钉死的上游字节在本机推导得到，因此比对仍在 `tiktoken.model` 层面
进行，而载入的对象是那份已认证的转换件。`toktier inspect` 仍是随包 family
列表的权威来源。

## 与现有工作的关系

Incremental BPE 研究的是 merge 阶段如何随新字节到来而增量扩展。TokTier
位于其上一层：会话状态保存完整 tokenizer 处理流程对应的 token ID 和
span，涵盖 normalization、pre-tokenization、merge 和 added-token 处理；
只有边界检查通过后，系统才会接受追加内容。

`llm-tokenizer` 和 NVIDIA Dynamo 的 `dynamo-tokenizers` 等服务项目也会
缓存编码结果。主要接口差异如下：

| 属性 | toktier | 进程内前缀缓存 |
|---|---|---|
| 生命周期 | 可持久化、跨进程 | 跟随 tokenizer 进程 |
| 命中验证 | 摘要用于筛选候选，再由已存字节验证 | 以摘要为键进行 lookup |
| 复用边界 | 经过认证的 tokenizer 边界 | 通常是 special-token 边界 |
| 接口形态 | 供自行持有会话状态的应用使用的 Python 库 | 服务网关组件 |

两层如何配合使用，见
[`docs/integration/dynamo.md`](docs/integration/dynamo.md)。

## 文档

- [`docs/releases/v0.2.6.md`](docs/releases/v0.2.6.md) — 本版本发布说明（英文）。
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

修正版 Gigatoken 是 11 个唯一 tokenizer 工件的默认认证 repair 窗口引擎；
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
