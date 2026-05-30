# CMGR 神经网络结构图

本文档按当前代码实现绘制，而不是只按论文示意图绘制。

关键代码入口：

- `cmgr_models/cmgr.py`
- `cmgr_models/depth_encoder.py`
- `cmgr_models/sagr.py`
- `cmgr_models/tam.py`
- `cmgr_models/bnd.py`

## 主网络前向结构

```mermaid
flowchart TD
    X["Point cloud<br/>[B, N, 3]"]

    X --> R["ReConEncoder<br/>3D point encoder<br/>frozen"]
    R --> RF["recon_final<br/>[B, 1536]"]
    R --> RI["recon_intermediates<br/>layers {0,4,8}<br/>[B, N3D, 384]"]
    RI --> RP["avg pool point tokens<br/>recon_pooled [B,384]"]

    X --> DE["DepthEncoder / CLIP2Point"]
    DE --> VS["ViewSelector + Renderer<br/>V views, current V=10"]
    VS --> DM["depth_maps<br/>[B,V,C,224,224]"]
    DM --> DV["CLIP ViT-B/32 visual<br/>inside DepthEncoder"]
    DV --> DF["depth_final<br/>[B*V,512]"]
    DV --> DI["depth_intermediates<br/>layers {0,4,8}<br/>[seq,B*V,768]"]

    DM --> TAM["TAM<br/>Texture Amplification Module"]
    RP --> TAM
    TAM --> EM["enhanced_maps<br/>[B,V,3,224,224]"]
    TAM --> LC["color_loss L_c<br/>via frozen CLIP image encoder"]

    RI --> SAGR["SAGR<br/>3D query + 2D key/value<br/>cross-attention + self-masking"]
    DI --> SAGR
    SAGR --> FU["rectified feature F_U<br/>[B*V,384]"]
    SAGR --> LMC["mask consistency loss L_mc"]

    RF --> AGG["Cross-view aggregation"]
    FU --> AGG
    DF --> AGG
    AGG --> FH["F_hat<br/>[B*V,512]"]
    FH --> POOL["mean over views"]
    POOL --> FHP["F_hat_pooled<br/>[B,512]"]

    CN["class names"] --> CT["Frozen CLIP text encoder"]
    CT --> FT["text features F_T<br/>[C,512]"]

    FHP --> GEO["geo logits<br/>100 * cos(F_hat, F_T)<br/>[B,C]"]
    FT --> GEO

    EM --> CIMG["Frozen CLIP image encoder<br/>in eval only"]
    CIMG --> CSIM["CLIP image-text logits<br/>pooled over views"]
    FT --> CSIM

    GEO --> TRAINLOGITS["training logits<br/>geo logits only"]
    GEO --> EVALSUM["eval logits<br/>geo logits + CLIP image-text logits"]
    CSIM --> EVALSUM

    TRAINLOGITS --> CE["cross entropy L_cls"]
    CE --> LOSS["total loss<br/>L_cls + alpha L_mc + beta L_c + gamma L_kd"]
    LMC --> LOSS
    LC --> LOSS
```

## 增量阶段 BND 路由结构

```mermaid
flowchart TD
    X["Point cloud<br/>[B,N,3]"]
    X --> R["ReConEncoder<br/>frozen"]
    R --> FP["point feature / recon_final<br/>[B,1536]"]

    FP --> BND["BND<br/>Linear 1536->256 + ReLU<br/>Linear 256->1"]
    BND --> LOGIT["raw logit"]
    LOGIT --> TH{"logit > h ?<br/>h = 0.1"}

    X --> NETB["NetB<br/>frozen base snapshot"]
    X --> NET["Current incremental net"]

    TH -->|"base"| NETB
    TH -->|"novel"| NET

    NETB --> PB["base prediction"]
    NET --> PN["novel/current prediction"]
```

## 参数训练状态

| 模块 | Base 阶段 | Incremental 阶段 | 说明 |
|---|---|---|---|
| ReConEncoder | 冻结 | 冻结 | 3D 点云特征提取器 |
| DepthEncoder / CLIP2Point visual | 训练 | 冻结 | 当前代码增量阶段冻结它 |
| CLIPWrapper text/image | 冻结 | 冻结 | 文本特征、颜色对齐和 eval image-text logits |
| SAGR | 训练 | 训练 | 跨模态几何校正 |
| TAM | 训练 | 训练 | 学习背景颜色并增强 depth maps |
| BND | 不训练 | 每个增量任务前单独训练 | base/novel 二分类路由 |

## 当前实现里的核心维度

| 张量 | 维度 |
|---|---|
| 输入点云 | `[B, N, 3]` |
| ReCon final | `[B, 1536]` |
| ReCon intermediate | `[B, N3D, 384]` |
| Depth maps | `[B, V, C, 224, 224]` |
| 当前 V | `10` |
| Depth final | `[B*V, 512]` |
| CLIP intermediate | `[seq, B*V, 768]` |
| SAGR output `F_U` | `[B*V, 384]` |
| Aggregated `F_hat` | `[B*V, 512]` |
| View pooled `F_hat` | `[B, 512]` |
| Text features | `[C, 512]` |
| Class logits | `[B, C]` |

