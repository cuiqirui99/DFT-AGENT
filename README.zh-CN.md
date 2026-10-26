# DFT AGENT

![DFT Agent：VASP、Quantum ESPRESSO、ABINIT、CP2K 和 Elk](docs/assets/dft-agent-banner.png)

**[下载 v0.4.7](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.7)** · [macOS](https://github.com/cuiqirui99/DFT-AGENT/releases/download/v0.4.7/DFT-Agent-0.4.7-macOS-arm64.dmg) · [Windows](https://github.com/cuiqirui99/DFT-AGENT/releases/download/v0.4.7/DFT-Agent-0.4.7-Windows-x64.exe) · [Python](https://pypi.org/project/dft-agent/0.4.7/) · [快速开始](docs/quickstart.md) · [English](README.md)

[Qirui Cui](https://www.kth.se/profile/qiruic?l=en) · KTH 皇家理工学院 · [ORCID](https://orcid.org/0009-0005-6165-3237)

在对话中描述计算任务，DFT Agent 为你准备输入文件，并在自己的超算集群上运行 VASP、Quantum ESPRESSO、ABINIT、CP2K 或 Elk。每个项目独立保存文件、对话和结果，可以在同一处修改任务、讨论数据。

![DFT Agent 工作流程](docs/assets/workflow.png)

支持结构编辑、弛豫、SCF、能带、态密度、磁性、SOC、DFT+U 和杂化泛函。[QE](docs/qe.md)还支持声子、NEB、电子–声子耦合和常规超导计算。[ABINIT、CP2K 和 Elk](docs/open-source-codes.md)提供声子、分子动力学和光学响应等方法。具体能力取决于计算程序。

## 安装

从 [v0.4.7](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.7) 下载：

- **Apple Silicon Mac，macOS 14+：**打开 `.dmg`，将 **DFT Agent** 拖入“应用程序”，再打开应用。
- **Windows x64：**运行 `.exe` 安装程序，然后从开始菜单打开 **DFT Agent**。

安装包内置 Python 和模型 SDK。Codex CLI、Claude Code CLI 为可选项，可以在应用里一键安装并开始登录。

也可使用 Python 3.11+ 从 [PyPI](https://pypi.org/project/dft-agent/0.4.7/) 安装：

```bash
python -m pip install "dft-agent[agent]==0.4.7"
dft-agent ui
```

[安装帮助](docs/install.md#desktop-app) · [从源码安装](docs/install.md#python-installation-from-source)。

## 开始使用

点击 **New project（新建项目）**，然后：

1. **连接 AI。** 打开 **Connections → Model（连接 → 模型）**，登录 Codex CLI 或 Claude Code CLI，也可以填写 API 密钥，再点击 **Connect**。[账号与 token 指南](docs/models.md)。
2. **连接超算。** 在 **Cluster（集群）** 输入 SSH 地址，按提示输入密码或验证码。选择已有计算程序，或在支持的集群上安装开源程序。[集群设置](docs/install.md#your-first-cluster-setup)。
3. **开始计算。** 在 **Files and runs → Files** 上传结构并点击 **Import files**，用自然语言描述任务并修改计划。点击 **Prepare inputs**，检查文件和资源后确认 **Submit**。

在 **Files and runs** 查看任务和下载输出，用 **Attach results** 选择要讨论的结果。工作流的不同步骤可以使用不同计算程序。

模型调用使用你自己的账号和额度。计算需要超算账号；VASP 需要已授权的程序和 POTCAR。[使用要求](docs/install.md#your-first-cluster-setup)。

## 离线试用

打开 **Settings → Classic workbench → Runs → Load sample runs**，无需连接超算即可浏览内置的 SrTiO3、Si、MgO 和 Fe 结果。手动准备输入和 Memory 浏览功能也保留在这里。

[快速开始](docs/quickstart.md) · [项目指南](docs/project-workspace.md#use-the-workspace) · [工作流程](docs/workflows.md) · [科学指导](docs/harness.md) · [记忆](docs/memory.md)

## 引用

Cui, Q. (2026). *DFT Agent* (v0.4.7). [源码](https://github.com/cuiqirui99/DFT-AGENT).

[v0.4.7 存档](https://doi.org/10.5281/zenodo.23664781)。

[安装](docs/install.md) · [Windows](docs/windows.md) · [方法](docs/methods.md) · [输入默认值](docs/input-defaults.md) · [命令行示例](docs/examples.md) · [VASP 验证](docs/validation-0.4.1.md) · [QE 验证](validation/qe-0.4.3/README.md) · [ABINIT/CP2K/Elk 检查](docs/validation-0.4.5.md) · [MIT 许可证](LICENSE) · [声明](NOTICE.md) · [引用](CITATION.cff)
