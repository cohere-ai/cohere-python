# Cohere Python SDK

![](banner.png)

[![version badge](https://img.shields.io/pypi/v/cohere)](https://pypi.org/project/cohere/)
![license badge](https://img.shields.io/github/license/cohere-ai/cohere-python)
[![fern shield](https://img.shields.io/badge/%F0%9F%8C%BF-SDK%20generated%20by%20Fern-brightgreen)](https://github.com/fern-api/fern)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

Cohere Python SDK 支持在多个主流云平台与环境中访问 Cohere 模型：包括 Cohere 官方平台、AWS（Bedrock、SageMaker）、Azure、GCP 以及 Oracle OCI。有关平台支持的完整清单及代码示例，请参阅 [SDK 支持文档页面](https://docs.cohere.com/docs/cohere-works-everywhere)。

## 文档指引 (Documentation)

Cohere 官方文档与完整的 API 参考请参阅[官方文档中心](https://docs.cohere.com/)。

## 安装指南 (Installation)

```bash
pip install cohere
```

## 快速上手 (Usage)

```python
import cohere

co = cohere.ClientV2()

response = co.chat(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

print(response)
```

> [!TIP]
> 您可以通过设置系统环境变量 `CO_API_KEY`，避免将 API Key 硬编码在代码中。例如，在 `~/.zshrc` 或 `~/.bashrc` 中添加：
> `export CO_API_KEY=您的账户API密钥`
> 保存后重新打开终端，调用 `cohere.Client()` 或 `cohere.ClientV2()` 的代码将自动读取此密钥。


## 流式输出 (Streaming)

本 SDK 支持流式接口。若要在对话中启用流式输出，请使用 `chat_stream`：

```python
import cohere

co = cohere.ClientV2()

response = co.chat_stream(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "hello world!"}],
)

for event in response:
    if event.type == "content-delta":
        print(event.delta.message.content.text, end="")
```

## Oracle 云基础设施 (OCI 支持)

本 SDK 原生支持 Oracle Cloud Infrastructure (OCI) 生成式 AI 服务。首先安装 OCI 扩展依赖：

```bash
pip install 'cohere[oci]'
```

然后使用 `OciClient` 或 `OciClientV2`：

```python
import cohere

# 使用 OCI 配置文件进行身份认证（默认读取 ~/.oci/config）
co = cohere.OciClient(
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)

response = co.embed(
    model="embed-english-v3.0",
    texts=["Hello world"],
    input_type="search_document",
)

print(response.embeddings)
```

### OCI 身份验证方式 (OCI Authentication Methods)

**1. 配置文件认证（默认方式）**
```python
co = cohere.OciClient(
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
    # 默认使用 ~/.oci/config 文件中的 DEFAULT 配置集
)
```

**2. 指定自定义 Profile**
```python
co = cohere.OciClient(
    oci_profile="MY_PROFILE",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**3. 基于会话的安全令牌认证 (Security Token)**
```python
# 适用于通过 OCI CLI 会话令牌生成的认证
co = cohere.OciClient(
    oci_profile="MY_SESSION_PROFILE",  # 配置了 security_token_file 的配置集
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**4. 直接传递凭据参数**
```python
co = cohere.OciClient(
    oci_user_id="ocid1.user.oc1...",
    oci_fingerprint="xx:xx:xx:...",
    oci_tenancy_id="ocid1.tenancy.oc1...",
    oci_private_key_path="~/.oci/key.pem",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

**5. 实例主体认证 (Instance Principal，用于 OCI 计算实例)**
```python
co = cohere.OciClient(
    auth_type="instance_principal",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1...",
)
```

### 支持的 OCI API (Supported OCI APIs)

OCI 客户端支持以下 Cohere 接口：
- **Embed（嵌入）**：全面支持所有向量嵌入模型
- **Chat（对话）**：全面支持 V1（`OciClient`）与 V2（`OciClientV2`）API
  - 支持通过 `chat_stream()` 获取流式响应
  - 支持 Command-R 与 Command-A 系列模型

### OCI 模型可用性与限制说明 (OCI Model Availability and Limitations)

**OCI 按需推理（On-Demand Inference）支持的功能：**
- ✅ **Embed 模型**：支持在 OCI Generative AI 上直接调用
- ✅ **Chat 模型**：支持通过 `OciClient` (V1) 与 `OciClientV2` (V2) 调用

**OCI 按需推理不支持的功能：**
- ❌ **Generate API**：OCI 中的 TEXT_GENERATION 属于基座模型，在部署前需要先进行微调
- ❌ **Rerank API**：OCI 中的 TEXT_RERANK 属于基座模型，在部署前需要先进行微调
- ❌ **多重 Embedding 类型**：OCI 按需模型单次请求仅支持单一嵌入类型（无法在单次请求中同时请求 `float` 与 `int8`）

**注意**：若要在 OCI 上使用 Generate 或 Rerank 模型，您需要：
1. 使用 OCI 的微调服务对基座模型进行微调
2. 将微调后的模型部署到专属终结点（Dedicated Endpoint）
3. 在代码中配置使用已部署的模型终结点

有关最新的模型可用性信息，请参阅 [OCI Generative AI 官方文档](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)。

## 参与贡献 (Contributing)

虽然我们非常重视开源社区对本 SDK 的贡献，但由于本项目的代码是由代码生成器自动生成的，直接对代码进行的修改必须同步迁移至我们的代码生成系统中，否则在下一次生成发布时会被自动覆盖。欢迎提交 PR 作为概念验证（PoC），但请注意我们无法直接合并未经生成器同步的代码改动。建议在修改代码前先提交 Issue 与我们讨论！

**另一方面，非常欢迎针对 README 文档的贡献与改进！**

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月19日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
