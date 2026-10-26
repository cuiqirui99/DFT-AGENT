# DFT Agent

![DFT Agent](docs/assets/dft-agent-logo.png)

版本 `0.4.7`。[English](README.md)

[Qirui Cui](https://www.kth.se/profile/qiruic?l=en) · KTH 皇家理工学院 · [ORCID](https://orcid.org/0009-0005-6165-3237)

用自然语言描述计算，检查输入文件，然后在自己的超算集群上运行 VASP、Quantum ESPRESSO、ABINIT、CP2K 或 Elk。每个项目独立保存文件、对话和结果，你可以反复修改任务、组合多种程序，并在计算结束后继续分析数据。

支持结构编辑、弛豫、SCF、能带、态密度、磁性、SOC、DFT+U 和杂化泛函，具体方法取决于计算程序。[QE](docs/qe.md)还支持安装部署、声子、NEB、电子–声子耦合和常规超导计算。[ABINIT、CP2K 和 Elk](docs/open-source-codes.md)可在支持的 Linux 集群上通过应用安装：ABINIT 支持声子，CP2K 支持大晶胞计算和分子动力学，Elk 支持光学和实时 TDDFT。

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

[从源码安装](docs/install.md#python-installation-from-source)。

## 开始使用

1. 点击 **New project（新建项目）**，为项目命名。
2. 打开 **Connections → Model（连接 → 模型）**。按提示安装并登录 **Codex CLI** 或 **Claude Code CLI**，也可以填写服务商的 API 密钥。已有连接可用 **Test connection** 检查。[模型设置](docs/models.md)。
3. 打开 **Cluster（集群）**，输入 `ssh 用户名@主机`。程序先尝试 SSH 密钥，再按集群要求询问密码或验证码。检查探测到的软件，或在 **Software（软件）** 中添加配置。缺少 QE 时，可在这里检查安装计划并启动构建。[集群设置](docs/install.md#your-first-cluster-setup)。
4. 回到对话，在 **Files and runs → Files → Add files** 上传结构，点击 **Import files**，然后描述任务。后续修改继续在同一对话中完成。
5. 点击 **Prepare inputs**，检查文件和计算资源，勾选确认框，再点击 **Submit**。在 **Files and runs → Jobs** 查看进度，在 **Results** 查看输出。

计算程序按工作流中的步骤选择，一个项目可以只用一种，也可以组合多种；无需切换整个项目。用 **Attach results** 选择要在对话中讨论的计算结果。

模型调用使用你自己的账号和额度。API 密钥及保存的 SSH 密码存入系统钥匙串，验证码不保存。VASP 需要已授权的程序和 POTCAR；QE、ABINIT、CP2K 和 Elk 都是开源软件，应用会自动安装它们及公开的赝势或基组。软件不包含模型额度或超算机时。

## 离线试用

打开 **Settings → Legacy workbench → Runs → Load sample runs**，可浏览内置的 SrTiO3、Si、MgO 和 Fe 计算结果。旧工作台保留手动准备输入、原有表单和 Memory 浏览功能。在 **Settings → Language / 语言** 中切换项目界面的语言。

[快速开始](docs/quickstart.md) · [项目指南](docs/project-workspace.md#use-the-workspace) · [工作流程](docs/workflows.md) · [科学指导](docs/harness.md) · [记忆](docs/memory.md)

## 引用

Cui, Q. (2026). *DFT Agent* (v0.4.7). [源码](https://github.com/cuiqirui99/DFT-AGENT).

[v0.4.7 存档](https://doi.org/10.5281/zenodo.23664781)。

[安装](docs/install.md) · [Windows](docs/windows.md) · [方法](docs/methods.md) · [输入默认值](docs/input-defaults.md) · [命令行示例](docs/examples.md) · [VASP 验证](docs/validation-0.4.1.md) · [QE 验证](validation/qe-0.4.3/README.md) · [ABINIT/CP2K/Elk 检查](docs/validation-0.4.5.md) · [MIT 许可证](LICENSE) · [声明](NOTICE.md) · [引用](CITATION.cff)
