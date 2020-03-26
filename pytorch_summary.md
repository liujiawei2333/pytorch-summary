# 配置

|               语句               |    解释     |
| :------------------------------: | :---------: |
|       `torch.__version__`        | pytorch版本 |
|       `torch.version.cuda`       |  CUDA版本   |
| `torch.backends.cudnn.version()` |  CUDNN版本  |
| `torch.cuda.get_device_name(0)`  |   GPU型号   |

**固定随机种子**

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

**程序运行在指定GPU上**

在代码中指定

```python
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
```

在命令行指定环境变量

```python
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

**判断是否有CUDA支持**

`torch.cuda.is_available()`

**cudnn**

`torch.backends.cudnn.benchmark = True`

提升卷积层的计算速度。

pytorch中的卷积层是有许多不同的实现算法的。对于不同的张量形状会有不同的最适用（最快）的算法。将其设置为True后在程序开始花费一点额外时间，让每个卷积层搜索最适合的实现算法。

由于一般的模型的结构都是确定的，所以都可以使用该设置。

**清除GPU存储**

`torch.cuda.empty_cache()`

或在nvidia-smi中找到程序的PID，用kill结束该进程

`kill -9 [pid]`

或直接重置没有被清空的GPU

`nvidia-smni –gpu-reset -i [gpu_id]`

# 张量数据

**张量基本信息**

|            语句            |                 解释                  |
| :------------------------: | :-----------------------------------: |
|       `input.type()`       |            input的数据类型            |
| `input.size()/input.shape` |        input的形状，类型为元组        |
|       `input.dim()`        |              input的维度              |
|       `input.item()`       |     获取只包含一个元素的张量的值+     |
|  `input.is_contiguous()`   | 布尔值，input内部元素在内存上是否连续 |

注：对张量进行按索引的切片操作后的张量与原张量共享内存，二者同时被改变

**张量数据类型**

| Data tyoe                | CPU tensor           | GPU tensor                |
| ------------------------ | -------------------- | ------------------------- |
| 32-bit floating point    | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
| 64-bit floating point    | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16-bit floating point    | N/A                  | `torch.cuda.HalfTensor`   |
| 8-bit integer (unsigned) | `torch.ByteTensor`   | `torch.cuda.ByteTensor`   |
| 8-bit integer (signed)   | `torch.CharTensor`   | `torch.cuda.CharTensor`   |
| 16-bit integer (signed)  | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
| 32-bit integer (signed)  | `torch.IntTensor`    | `torch.cuda.IntTensor`    |
| 64-bit integer (signed)  | `torch.LongTensor`   | `torch.cuda.LongTensor`   |

**张量创建**

|                 语句                  |                             解释                             |
| :-----------------------------------: | :----------------------------------------------------------: |
|          `torch.rand(sizes)`          |               [0,1)之间的均匀分布抽取的随机数                |
|       `torch.rand_like(input)`        |               与input形状相同的张量，rand分布                |
|         `torch.randn(sizes)`          |         标准正态分布（均值为0，方差为1）抽取的随机数         |
|       `torch.randn_like(input)`       |               与input形状相同的张量，randn分布               |
|       `torch.normal(means,std)`       | 离散正态分布（means,std）中抽取的随机数<br>std是张量，包含每个输出元素相关的标准差 |
|    `torch.randint(low,high,size)`     |       均匀分布的[low,high]之间的整数随机值，尺寸为size       |
| `torch.randint_like(input,low,high)`  |              与input形状相同的张量，randint分布              |
|   `torch.linspace(start,end,steps)`   |       1维张量，包含区间start和end上均匀间隔的step个点        |
| `torch.randperm(n,dtype=torch.int64)` |                     0~n-1之间的随机排列                      |
|          `torch.ones(sizes)`          |                       元素全是1的张量                        |
|        `torch.ones_like(size)`        |               与input形状相同的张量，元素全是1               |
|         `torch.zeros(sizes)`          |                       元素全是0的张量                        |
|       `torch.zeros_like(input)`       |               与input形状相同的张量，元素全为0               |

**张量操作**

|               命令               |                             解释                             |
| :------------------------------: | :----------------------------------------------------------: |
|      `input[input < 1] = 0`      |          input中小于1的元素赋值为0，不创建新的张量           |
|     `torch.where(x < 1,a,b)`     | x小于1的元素赋值为a，否则赋值为b，其中a,b的形状与x相同，创建新的张量 |
| `torch.clamp(input,min=a,max=b)` | input中小于a的元素全赋值为1，大于b的元素全赋值为b，中间的元素不变 |
|        `input.flatten()`         |                    按行优先展开成一维向量                    |
|                                  |                                                              |

**数据类型转换**

|       命令       |            解释             |
| :--------------: | :-------------------------: |
|  `input.cuda()`  |          放在GPU上          |
|  `input.cpu()`   |          放在CPU上          |
| `input.float()`  |    转换为float(32bit）型    |
|  `input.long()`  |        转换为长整型         |
|  `input.half()`  | 转换为半精度浮点（16bit）型 |
| `input.double()` | 转换为双精度浮点（64bit）型 |
|  `input.byte()`  |       转换为8bit整型        |

**与numpy转换**

|               命令                |       解释        |
| :-------------------------------: | :---------------: |
|       `input.cpu().numpy()`       | tensor转换为numpy |
| `torch.from_numpy(input).float()` | numpy转换为tensor |

**维度转换**

|             命令              |                             解释                             |
| :---------------------------: | :----------------------------------------------------------: |
|    `input.permute(1,2,0)`     |          对张量的任意维度转置，由(C,H,W)变为(H,W,C)          |
| `torch.transpose(input, 3,1)` |                    对张量的某两个维度转置                    |
|       `input.view(a,b)`       | 按行排列为a*b形状的张量；当其中一个参数为-1时，即为需要推断出的参数；不创建新的对象 |
| `torch.reshape(input,shape)`  |         作用与view相同，但是能够处理张量不连续的情况         |
|     `input.contiguous()`      |               重新开辟一块内存连续地存储input                |

**与PIL.Image转换**

```python
#tensor->PIL.Image
image = PIL.Image.fromarray(torch.clamp(input * 255,min=0,max=255).byte().permute(1,2,0).cpu().numpy())
或者
image = torchvision.transforms.fuctional.to_pil_image(input)

#PIL.Image->tensor
input = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
或者
input = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))
```

**np.ndarray与PIL.Image转换**

```python
# np.ndarray -> PIL.Image.
image = PIL.Image.fromarray(ndarray.astypde(np.uint8))

# PIL.Image -> np.ndarray.
ndarray = np.asarray(PIL.Image.open(path))
```

**张量复制**

|           命令           | 在计算图中 |  共享内存  |
| :----------------------: | :--------: | :--------: |
|     `input.clone()`      |     是     | 开辟新内存 |
|     `input.detach()`     |     否     |  共享内存  |
| `input.detach().clone()` |     否     | 开辟新内存 |

**拼接张量**

|                       命令                       |      解释      |
| :----------------------------------------------: | :------------: |
|  `input = torch.cat((input1,input2,...),dim=0)`  | 沿给定维度拼接 |
| `input = torch.stack((input1,input2,...),dim=0)` | 增加一个新维度 |

