## 1. 论文介绍

### 1.1 摘要

- 本文质疑了基于Transformer的长期时间序列预测（LTSF）解决方案的有效性。
- 尽管Transformer在序列建模中非常成功，但在时间序列建模中，我们更关注连续点集的时间关系，而非元素间的语义关联。
- 作者提出了一个简单的一层线性模型（LTSF-Linear），用于与现有的复杂的基于Transformer的LTSF模型进行比较。
- 实验结果表明，LTSF-Linear在所有情况下都意外地优于现有的基于Transformer的LTSF模型，并且通常优势明显。

### 1.2 贡献

- 首次对基于Transformer的长期时间序列预测任务的有效性提出质疑。
- 引入了LTSF-Linear模型，并在九个基准数据集上与现有的基于Transformer的LTSF解决方案进行了比较。
- 进行了全面的实证研究，探讨了现有Transformer-based TSF解决方案的各个方面，包括对长期输入的建模能力、对时间序列顺序的敏感性、位置编码和子序列嵌入的影响，以及效率比较。

### 1.3 实验设置

- **数据集：**使用了九个真实世界的数据集，涵盖了交通、能源、经济、天气和疾病预测等不同应用领域。

  - **ETT (Electricity Transformer Temperature)**: 包含两个按小时记录的数据集（ETTh）和两个15分钟记录的数据集（ETTm）。每个数据集包含了从2016年到2018年的电力变压器的七个油和负载特性。
  - **Traffic**: 描述了道路占用率，包含了从2015年到2016年旧金山高速公路上传感器记录的每小时数据。
  - **Electricity**: 收集了从2012年到2014年321个客户的每小时电力消耗数据。
  - **Exchange-Rate**: 收集了从1990年到2016年8个国家的每日汇率数据。
  - **Weather**: 包括21个天气指标，如气温和湿度，数据每10分钟记录一次，涵盖了2020年德国的天气情况。
  - **ILI (Influenza-Like Illness)**: 描述了流感样疾病患者比例和患者总数，包含了从2002年到2021年美国疾病控制与预防中心的每周数据。
  - **ETTh1**: ETT数据集的一个子集，按小时记录电力变压器油和负载特性。
  - **ETTh2**: ETT数据集的另一个子集，与ETTh1类似，但可能有不同的特性或时间段。
  - **ETTm1** 和 **ETTm2**: 分别为15分钟记录级别的ETT数据集的两个子集，包含了电力变压器的相关特性。

- **评价指标：**使用均方误差（MSE）和平均绝对误差（MAE）作为核心评价指标。
  $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 
  $$
  
  $$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |Y_i - \hat{Y}_i| 
  $$
  

### 1.4 实验结果

- LTSF-Linear在多变量预测中超过了最先进的FEDformer模型，改进幅度在20%到50%之间。
- 对于不同的时间序列基准，NLinear和DLinear显示出处理分布变化和趋势-季节性特征的优越性。
- LTSF-Linear在单变量预测中也一致地优于基于Transformer的LTSF解决方案。

### 1.5 结论

- 作者得出结论，现有的基于Transformer的时间序列建模能力被夸大了，至少对于现有的LTSF基准测试而言。
- LTSF-Linear作为一个简单的基线，为未来关于具有挑战性的长期时间序列预测问题的研究提供了一个起点。

## 2. pytorch实现版本

### 2.1 准备工作

创建环境：

```
conda create -n mindspore python=3.9
```

安装依赖包：

```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
asttokens                 2.4.1                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
ca-certificates           2024.3.11            h06a4308_0  
contourpy                 1.2.1                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
fonttools                 4.53.0                   pypi_0    pypi
importlib-resources       6.4.0                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.4.4                h6a678d5_1  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
matplotlib                3.9.0                    pypi_0    pypi
mindspore                 2.2.14                   pypi_0    pypi
msadapter                 0.1.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
numpy                     1.26.4                   pypi_0    pypi
openssl                   3.0.13               h7f8727e_2  
packaging                 24.0                     pypi_0    pypi
pandas                    2.2.2                    pypi_0    pypi
pillow                    10.3.0                   pypi_0    pypi
pip                       24.0             py39h06a4308_0  
protobuf                  5.27.1                   pypi_0    pypi
psutil                    5.9.8                    pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
python                    3.9.19               h955ad1f_1  
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2024.1                   pypi_0    pypi
readline                  8.2                  h5eee18b_0  
scikit-learn              1.5.0                    pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
setuptools                69.5.1           py39h06a4308_0  
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
threadpoolctl             3.5.0                    pypi_0    pypi
tk                        8.6.14               h39e8969_0  
typing-extensions         4.12.1                   pypi_0    pypi
tzdata                    2024.1                   pypi_0    pypi
wheel                     0.43.0           py39h06a4308_0  
xz                        5.4.6                h5eee18b_1  
zipp                      3.19.2                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_1  

```

数据集中所有csv文件放入dataset目录下

项目目录树结构如下：
```
F:.
│  .gitignore
│  LICENSE
│  LTSF-Benchmark.md
│  README.md
│  requirements.txt
│  result.txt
│  run_longExp.py
│  run_stat.py
│  weight_plot.py
│
├─dataset
│      electricity.csv
│      ETTh1.csv
│      ETTh2.csv
│      ETTm1.csv
│      ETTm2.csv
│      exchange_rate.csv
│      national_illness.csv
│      traffic.csv
│      weather.csv
│
├─data_provider
│     data_factory.py
│     data_loader.py
│     __init__.py
│
├─exp
│     exp_basic.py
│     exp_main.py
│     exp_stat.py
│  
│
├─FEDformer
│  │  LICENSE
│  │  README.md
│  │  run.py
│  │
│  ├─data_provider
│  │      data_factory.py
│  │      data_loader.py
│  │
│  ├─exp
│  │      exp_basic.py
│  │      exp_main.py
│  │
│  ├─layers
│  │      AutoCorrelation.py
│  │      Autoformer_EncDec.py
│  │      Embed.py
│  │      FourierCorrelation.py
│  │      MultiWaveletCorrelation.py
│  │      SelfAttention_Family.py
│  │      Transformer_EncDec.py
│  │      utils.py
│  │
│  ├─models
│  │      Autoformer.py
│  │      FEDformer.py
│  │      Informer.py
│  │      Transformer.py
│  │
│  ├─scripts
│  │      LongForecasting.sh
│  │      LookBackWindow.sh
│  │
│  └─utils
│          masking.py
│          metrics.py
│          timefeatures.py
│          tools.py
├─layers
│     AutoCorrelation.py
│     Autoformer_EncDec.py
│     Embed.py
│     SelfAttention_Family.py
│     Transformer_EncDec.py
├─models
│     Autoformer.py
│     DLinear.py
│     Informer.py
│     Linear.py
│     NLinear.py
│     Stat_models.py
│     Transformer.py
│
├─pics
│  │  efficiency.png
│  │  Linear.png
│  │  Mul-results.png
│  │  Uni-results.png
│  │  Visualization_DLinear.png
│  │
│  └─DLinear
│          .DS_Store
│          DLinear.png
│          DLinear_results.png
│          DLinear_Univariate_Results.png
│          LookBackWindow.png
│          results.png
│
├─Pyraformer
│  │  data_loader.py
│  │  LEGAL.md
│  │  LICENSE
│  │  long_range_main.py
│  │  preprocess_elect.py
│  │  preprocess_flow.py
│  │  preprocess_wind.py
│  │  README.md
│  │  requirements.txt
│  │  simulate_sin.py
│  │  single_step_main.py
│  │
│  ├─pyraformer
│  │  │  embed.py
│  │  │  graph_attention.py
│  │  │  hierarchical_mm_tvm.py
│  │  │  Layers.py
│  │  │  Modules.py
│  │  │  PAM_TVM.py
│  │  │  Pyraformer_LR.py
│  │  │  Pyraformer_SS.py
│  │  │  SubLayers.py
│  │  │
│  │  └─lib
│  │          lib_hierarchical_mm_float32_cuda.so
│  │
│  ├─scripts
│  │      LongForecasting.sh
│  │      LookBackWindow.sh
│  │
│  └─utils
│      │  timefeatures.py
│      │  tools.py
│      │
│      └─__pycache__
│              timefeatures.cpython-37.pyc
│              tools.cpython-37.pyc
│
├─scripts
│  ├─EXP-Embedding
│  │      Formers_Embedding.sh
│  │
│  ├─EXP-LongForecasting
│  │  │  Formers_Long.sh
│  │  │  Linear-I.sh
│  │  │  Stat_Long.sh
│  │  │
│  │  └─Linear
│  │      │  .DS_Store
│  │      │  electricity.sh
│  │      │  etth1.sh
│  │      │  etth2.sh
│  │      │  ettm1.sh
│  │      │  ettm2.sh
│  │      │  exchange_rate.sh
│  │      │  ili.sh
│  │      │  traffic.sh
│  │      │  weather.sh
│  │      │
│  │      └─univariate
│  │              etth1.sh
│  │              etth2.sh
│  │              ettm1.sh
│  │              ettm2.sh
│  │
│  └─EXP-LookBackWindow
│          Formers_LookBackWindow.sh
│          Linear_DiffWindow.sh
│
└─utils
       masking.py
       metrics.py
       timefeatures.py
       tools.py
     
    
```

### 2.2 运行代码

在项目目录下执行
```
‘sh scripts/EXP-LongForecasting/Linear/exchange_rate.sh’

```

## 3. mindspore实现版本

代码仓库：https://github.com/XiShuFan/MAMO_mindspore

## 3.1 mindspore框架介绍

MindSpore是华为推出的一款人工智能计算框架，主要用于开发AI应用和模型。它的特点如下:

- 框架设计：MindSpore采用静态计算图设计，PyTorch采用动态计算图设计。静态计算图在模型编译时确定计算过程，动态计算图在运行时确定计算过程。静态计算图通常更高效，动态计算图更灵活；
- 设备支持：MindSpore在云端和边缘端都有较好的支持，可以在Ascend、CPU、GPU等硬件上运行；
- 自动微分：MindSpore提供自动微分功能，可以自动求导数，简化模型训练过程；
- 运算符和层：MindSpore提供丰富的神经网络层和运算符，覆盖CNN、RNN、GAN等多种模型；
- 训练和部署：MindSpore提供方便的模型训练和部署功能，支持ONNX、CANN和MindSpore格式的模型导出，可以部署到Ascend、GPU、CPU等硬件；



### 3.2 环境准备

操作系统Ubuntu 20.04, Anaconda3, Python3.9
创建虚拟环境并且切换到环境：

```
conda create -n mindspore python=3.9
conda activate mindspore
```

克隆已经实现好的mindspore版本LTSF-Linear代码：

```
git clone https://github.com/XiShuFan/MAMO_mindspore.git
```

安装mindspore与必要依赖包：

```
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install msadapter
```


### 3.3 模型迁移

将Pytorch的API替换成mindspore的API，官方给出了[文档说明](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)。
下面是我在模型迁移过程替换的API以及Class：

| pytorch API / Class       | mindspore API/Class                | 说明                          | 两者差异                                                     |
| ------------------------- | ---------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| torch.from_numpy          | mindspore.tensor.from_numpy        | 从numpy得到tensor             | 无                                                           |
| torch.tensor.to           | mindspore.tensor.to_device         | 将tensor传入指定的设备        | 无                                                           |
| torch.zeros_like          | mindspore.ops.ZerosLike            | 获得指定shape的全零元素tensor | 无                                                           |
| torch.nn.Sequential       | mindspore.nn.SequentialCell        | 整合多个网络模块              | 无                                                           |
| torch.mean                | mindspore.ops.ReduceMean           | 计算均值                      | 无                                                           |
| torch.optim.Adam          | mindspore.nn.Adam                  | 优化器                        | 无                                                           |
| torch.nn.Module           | mindspore.nn.Cell                  | 神经网络的基本构成单位        |                                                              |
| torch.nn.Linear           | mindspore.nn.Dense                 | 全连接层                      | PyTorch：全连接层，实现矩阵相乘的运算。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 |
| torch.cat                 | mindspore.ops.concat               | tensor按照指定维度拼接        | 无                                                           |
| torch.tensor.view         | mindspore.ops.Reshape              | 重新排列tensor的维度          | 无                                                           |
| Adam.zero_grad            | Adam.clear_grad                    | 清除梯度                      | 无                                                           | 

对data_provider和model中的python文件，使用MSAdapter对pytorch代码进行mindspore迁移
```python
# import torch
# import torch.nn as nn
# import torchvision import datasets, transforms

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import mindspore as ms
from msadapter.pytorch.utils.data import DataLoader, Dataset
```

### 3.4 详细迁移代码

#### 定义训练

```python
...
import mindspore
...
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        def forward_fn(batch_x, batch_y):
            outputs = self.model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(outputs, batch_y)
            return loss, outputs

        def train_step(batch_x, batch_y):
            (loss, outputs), grad = grad_fn(batch_x, batch_y)
            model_optim(grad)
            return loss, outputs

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                grad_fn = ms.value_and_grad(forward_fn, None, model_optim.parameters, has_aux=True)

                loss, outputs= train_step(batch_x, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
```



#### 网络实现

```python
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
```


### 3.5 训练结果

模型在electricity数据上预测96步的结果：

```
>>>>>>>testing : Electricity_336_96_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 5165
mse:0.23239946365356445, mae:0.3386293947696686
```

