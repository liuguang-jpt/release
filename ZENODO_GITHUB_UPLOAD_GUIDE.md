# GitHub 与 Zenodo 正式发布操作清单（release v1.0.0）

> 状态：本地发布候选包已经整理；GitHub 仓库尚未创建，Zenodo DOI `10.5281/zenodo.22015283` 为作者提供的目标/预留标识。只有在仓库、GitHub Release 和 Zenodo 记录均已公开且可访问后，论文才能写成“publicly available”。

## 一、发布源目录

只从本地发布候选包目录发布，不要从整个科研项目根目录上传。公开仓库中不要写入本机绝对路径。

```text
<local release source directory>
```

## 二、GitHub 新建仓库时填写

打开 GitHub 的“New repository”，填写：

```text
Owner: liuguang-jpt
Repository name: release
Description: Bias-aware PROTAC degradation benchmark: derived metadata, reproducible code and external-validation rebuild scripts
Visibility: Public
```

以下三项均不要勾选，因为本地发布包已经包含对应文件：

```text
Add a README file: 不勾选
Add .gitignore: 不勾选
Choose a license: 不选择
```

创建后，目标地址应为：

```text
https://github.com/liuguang-jpt/release
```

## 三、GitHub 应上传的内容

将发布源目录的以下内容作为仓库根目录上传：

```text
.gitignore
README.md
LICENSE.md
CITATION.cff
requirements.txt
requirements-optional-torch.txt
ZENODO_GITHUB_UPLOAD_GUIDE.md
code/
external_code/
data/derived/
data/raw/raw_data_manifest.csv
data/external/
docs/
```

其中 `data/external/` 只能包含聚合指标和模型清单，不得出现 TPDdb 记录级队列、标签或逐记录预测。

### 禁止上传

```text
PROTAC-DB 原始快照
TPDdb 原始文件
TPDdb 记录级外部队列
TPDdb 记录级标签
TPDdb 逐记录预测
PROTAC-PatentDB 受限数据
__pycache__/
*.pyc
虚拟环境、模型缓存和本机日志
作者私有材料和内部 AI 工作底稿
_B_workspace_raw.csv
*raw_submission.xlsx
任何包含本机用户目录、项目根目录或其他机器专属路径的文件
```

## 四、Git 推送命令

在发布源目录中执行：

```bash
git init
git branch -M main
git add -A
git commit -m "Release v1.0: bias-aware PROTAC benchmark and rebuild scripts"
git remote add origin https://github.com/liuguang-jpt/release.git
git push -u origin main
```

随后创建论文冻结标签：

```bash
git tag -a v1.0-paper -m "Version corresponding to the manuscript release"
git push origin v1.0-paper
```

## 五、在 GitHub 页面创建正式 Release

仅有 tag 不足以触发标准归档流程。进入仓库页面的 Releases，创建：

```text
Tag: v1.0-paper
Release title: v1.0.0 — manuscript release
Target: main
Set as the latest release: 是
Pre-release: 否
```

建议 Release notes：

```text
First manuscript-aligned release of the bias-aware PROTAC degradation benchmark.

Contents:
- reproducible ETL, labelling, grouped-split and modelling code;
- PROTAC-DB-derived benchmark artefacts subject to upstream terms;
- a 132-record adjudicated internal-consistency sample (not independent expert validation);
- aggregate TPDdb-derived external-evaluation outputs and rebuild scripts;
- mixed-licence and provenance documentation.

Raw PROTAC-DB snapshots and TPDdb-derived record-level cohorts, labels and predictions are not redistributed.
```

发布后记录 GitHub 自动生成的 Source code ZIP，并计算其 SHA-256。

## 六、Zenodo：先确认 DOI 对应的记录

作者提供的 DOI：

```text
10.5281/zenodo.22015283
```

先登录 Zenodo，检查该 DOI 是否已经对应你的草稿、预留记录或已发布记录：

1. 若它对应现有草稿/预留记录：继续编辑该记录，不要再通过 GitHub–Zenodo 集成创建第二条记录。
2. 若它已发布：检查版本和文件是否与 GitHub `v1.0-paper` 完全一致；不一致时应按 Zenodo 版本机制更新，不要覆盖历史版本。
3. 若账号中完全找不到该记录且 DOI 不能解析：先确认 DOI 是否抄写正确，再决定是否建立新记录。

## 七、Zenodo 上传什么文件

### 推荐：复用作者已有的 DOI 草稿

从 GitHub `v1.0-paper` Release 下载或生成一个不含 `.git/` 的归档：

```text
bias-aware-protac-benchmark-v1.0.0.zip
SHA256SUMS.txt
```

ZIP 内应与 GitHub tag 对应的仓库内容一致。`SHA256SUMS.txt` 至少记录 ZIP 的 SHA-256。不要额外上传原始数据库、TPDdb 记录级文件或本机日志。

### 仅当没有预留 DOI 时

可启用 GitHub–Zenodo 集成，再发布 GitHub Release，由 Zenodo 自动抓取归档。若已经为 `10.5281/zenodo.22015283` 预留记录，不要同时启用自动抓取来创建重复 DOI。

## 八、Zenodo 元数据逐项填写

### Upload type / Resource type

```text
Software
```

如界面支持组合资源类型，可选择“Dataset and software”；否则使用 Software，并在描述中明确包含派生基准 artefacts 和代码。

### Title

```text
Bias-Aware Learning from Censored and Incompletely Observed PROTAC Degradation Data: Dataset and Code
```

### Creators

```text
1. Liu, Guanglu
   ORCID: 0009-0006-8693-1923
   Affiliation: China University of Petroleum (East China)

2. Wang, Shuang
   Affiliation: China University of Petroleum (East China)
   Email: wangshuang@upc.edu.cn
```

若王爽老师有 ORCID，在发布前补入；没有时不要虚构。

### Version

```text
1.0.0
```

### Publication date

填写 Zenodo 实际公开发布当天的日期，不要提前填造发布日期。

### Description

```text
This release accompanies the manuscript “Bias-Aware Learning from Censored and Incompletely Observed PROTAC Degradation Data”. It contains project-authored code and documentation for reproducible ETL, operational P/N/A/U evidence labelling, grouped validation, positive–unlabelled learning, probability calibration, censoring-aware evaluation and sensitivity analyses. It also contains PROTAC-DB-derived benchmark artefacts, a 150-record calibration annotation set, a 132-record adjudicated internal-consistency sample, and aggregate outputs from a structure-disjoint TPDdb-derived external evaluation.

The 132-record sample assesses protocol consistency and is not an independent expert gold standard. The TPDdb-derived evaluation is bounded external evidence; it is not prospective, laboratory-independent or fully time-independent validation.

Original code is licensed under the MIT License. Project-authored documentation and annotation protocols are licensed under CC BY 4.0. Record-level PROTAC-DB-derived materials remain subject to upstream terms, and no additional rights are granted by this deposit. Raw PROTAC-DB snapshots and TPDdb-derived record-level cohort data, labels and predictions are not redistributed.

Repository: https://github.com/liuguang-jpt/release
Manuscript tag: v1.0-paper
```

### Keywords

逐个填写：

```text
PROTAC
targeted protein degradation
positive-unlabeled learning
censored data
label bias
calibration
grouped validation
benchmark
reproducibility
external validation
```

### Licence

此包是混合许可，不能笼统选择 CC BY 4.0。若 Zenodo 只能选择一个主许可：

```text
Other / Custom（选择界面中最接近的“其他/自定义”项）
```

并在 Description 与 Notes 中保留以下说明：

```text
Original code: MIT License.
Project-authored documentation and annotation protocols: CC BY 4.0.
PROTAC-DB-derived record-level materials: subject to upstream terms; no additional rights granted.
TPDdb-derived record-level cohort data, labels and predictions: not redistributed.
```

### Related/alternate identifiers

```text
Related identifier: https://github.com/liuguang-jpt/release
Relation: Is supplemented by / Is identical to（按界面可用关系选择；若 ZIP 与 tag 完全一致，优先 Is identical to）
```

如果可填写 GitHub Release URL，应在 Release 创建后使用 `v1.0-paper` 的正式 Release 地址。

### Access right

```text
Open Access
```

前提是上传包中没有被禁止再分发的原始/记录级数据。

### Funding

与论文 Funding 声明保持完全一致；不得临时新增项目号。若论文声明无专项资助，则按 Zenodo 界面留空并在论文中保持一致。

## 九、发布前后核验

发布前：

- [ ] GitHub 仓库为 Public。
- [ ] `v1.0-paper` tag 已推送。
- [ ] GitHub Release 已正式发布，不是 Draft。
- [ ] ZIP 与 tag 内容一致。
- [ ] 不含本机绝对路径、缓存、私有文件和禁止再分发的数据。
- [ ] Zenodo 文件 SHA-256 已记录。
- [ ] 通讯作者书面同意公开。

发布后：

- [ ] `https://github.com/liuguang-jpt/release` 可在未登录状态访问。
- [ ] GitHub `v1.0-paper` Release 可下载。
- [ ] DOI `10.5281/zenodo.22015283` 可解析并显示正确标题、作者、版本和文件。
- [ ] Zenodo 文件与 GitHub tag 一致。
- [ ] 再将论文 Data Availability 和 Code Availability 从“prepared/pending”升级为“publicly available”。
- [ ] 若最终 DOI 与作者提供的 DOI 不同，同步更新论文 Markdown、Word、README 和 `CITATION.cff`。
