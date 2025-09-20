使用流程：
1.打开终端 (Anaconda Prompt 或 PowerShell),启动您位于F盘的 FaultD 环境的命令是使用它的完整路径来激活。

启动命令如下：
Bash
conda activate F:\Anaconda3\envs\FaultD

2.启动 Visdom server,并保持这个终端窗口不要关闭，让它在后台持续运行。：

Bash
python -m visdom.server

3.另外打开一个新终端 (Anaconda Prompt, PowerShell, VS code等)，启动您位于F盘的 FaultD 环境

启动命令如下：
Bash
conda activate F:\Anaconda3\envs\FaultD

4.在终端里运行您的主程序：

Bash
python run_launcher.py

