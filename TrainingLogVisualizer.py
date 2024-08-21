# 导入所需库
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib import colors as mcolors

# 设置Matplotlib中文字体
import matplotlib
matplotlib.rc("font", family='SimHei')  # 中文字体

# 切换工作目录到MMSegmentation主目录
os.chdir(r'mmsegmentation-main')
print("Current working directory:", os.getcwd())

# 载入训练日志
log_path = r"scalars.json"
with open(log_path, "r") as f:
    json_list = f.readlines()

# 初始化两个空的DataFrame
df_train = pd.DataFrame()
df_test = pd.DataFrame()

# 创建两个列表来收集数据
train_data = []
test_data = []

for each in json_list[:-1]:
    if 'aAcc' in each:
        test_data.append(eval(each))
    else:
        train_data.append(eval(each))

# 将收集的数据转换为DataFrame
df_train = pd.concat([df_train, pd.DataFrame(train_data)], ignore_index=True)
df_test = pd.concat([df_test, pd.DataFrame(test_data)], ignore_index=True)

# 导出训练日志表格
df_train.to_csv('训练日志-训练集.csv', index=False)
df_test.to_csv('训练日志-测试集.csv', index=False)

# 可视化辅助函数
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 
          'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 
          'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 
          'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 
          'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 
          'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 
          'deeppink', 'hotpink']
linestyle = ['--', '-.', '-']

def get_line_arg():
    """随机产生一种绘图线型"""
    line_arg = {
        'color': random.choice(colors),
        'linestyle': random.choice(linestyle),
        'linewidth': random.randint(1, 4)
    }
    return line_arg

# 训练集损失函数可视化
metrics = ['loss', 'decode.loss_ce', 'aux.loss_ce']
plt.figure(figsize=(16, 8))

x = df_train['step']
for y in metrics:
    try:
        plt.plot(x, df_train[y], label=y, **get_line_arg())
    except KeyError:
        pass

plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('训练集损失函数', fontsize=25)
plt.legend(fontsize=20)
plt.savefig('训练集损失函数.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 训练集准确率可视化
metrics = ['decode.acc_seg', 'aux.acc_seg']
plt.figure(figsize=(16, 8))

x = df_train['step']
for y in metrics:
    try:
        plt.plot(x, df_train[y], label=y, **get_line_arg())
    except KeyError:
        pass

plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.title('训练集准确率', fontsize=25)
plt.legend(fontsize=20)
plt.savefig('训练集准确率.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 测试集评估指标可视化
metrics = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall']
plt.figure(figsize=(16, 8))

x = df_test['step']
for y in metrics:
    try:
        plt.plot(x, df_test[y], label=y, **get_line_arg())
    except KeyError:
        pass

plt.tick_params(labelsize=20)
plt.ylim([0, 100])
plt.xlabel('step', fontsize=20)
plt.ylabel('Metrics', fontsize=20)
plt.title('测试集评估指标', fontsize=25)
plt.legend(fontsize=20)
plt.savefig('测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')
plt.show()
