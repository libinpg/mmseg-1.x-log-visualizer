# mmseg-1.x-log-visualizer
A tool for visualizing training logs generated in MMSegmentation 1.x framework.
参考链接：[同济子豪兄](https://github.com/TommyZihao/MMSegmentation_Tutorials/blob/main/20230816/%E3%80%90H1%E3%80%91%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97-%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E6%80%BB%E4%BD%93%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.ipynb)


# mmseg-1.x-log-visualizer

## 简介

`TrainingLogVisualizer.py` 是一个用于可视化深度学习训练日志的 Python 脚本。它可以解析由 MMSegmentation 等框架生成的日志文件，并生成训练过程中的损失函数、准确率和评估指标的图表。

## 特性

- 解析并可视化训练集和测试集的日志数据。
- 支持展示训练过程中的损失函数、准确率、IoU、精确率、召回率等多种评估指标。
- 自动保存生成的图表为 PDF 格式。

## 环境依赖

运行本脚本需要以下 Python 包：

- Python 3.7+
- pandas
- matplotlib

## 使用说明

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/TrainingLogVisualizer.git
cd TrainingLogVisualizer
```

### 2. 安装依赖

```bash
pip install pandas matplotlib
```

### 3. 配置中文字体（可选）

如果在图表中需要支持中文显示，请下载并配置 SimHei 字体。以 Linux 操作为例：

```bash
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /path/to/matplotlib/fonts/SimHei.ttf
rm -rf ~/.cache/matplotlib
```

修改 `TrainingLogVisualizer.py` 中的字体配置：

```python
import matplotlib
matplotlib.rc("font", family='SimHei')  # 中文字体
```

### 4. 修改参数

脚本中的一些路径和配置需要根据你的实际情况进行修改：

- **日志文件路径**：修改脚本中 `log_path` 变量的值，指向你的 JSON 日志文件。例如：

  ```python
  log_path = './work_dirs/ZihaoDataset-PSPNet/20230818_210528/vis_data/scalars.json'
  ```

- **输出图表路径**：脚本默认会将生成的图表保存为 PDF 文件在 `图表/` 目录下。如果需要更改输出路径，可以在 `plt.savefig` 中修改路径参数：

  ```python
  plt.savefig('图表/训练集损失函数.pdf', dpi=120, bbox_inches='tight')
  ```

### 5. 运行脚本

确保日志文件路径正确，运行脚本：

```bash
python TrainingLogVisualizer.py
```

### 6. 查看输出

脚本将生成多个 PDF 文件，分别展示训练过程中的各项指标：

- `训练集损失函数.pdf`
- `训练集准确率.pdf`
- `测试集分类评估指标.pdf`

## 日志文件格式

脚本支持从 MMSegmentation 等框架生成的 JSON 日志文件中提取数据，解析并生成可视化图表。

