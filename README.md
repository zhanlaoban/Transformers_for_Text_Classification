# Transformers_for_Text_Classification

# 基于Transformers的文本分类

基于最新的 [huggingface](https://github.com/huggingface) 出品的 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) v2.2.2代码进行重构。为了保证代码日后可以直接复现而不出现兼容性问题，这里将 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) 放在本地进行调用。



# Highlights

- 支持transformer模型后接各种特征提取器
- 支持测试集预测代码
- 精简原始transformers代码，使之更适合文本分类任务
- 优化logging终端输出，使之输出内容更加合理



# Support 

**model_type：**

- [x] bert
- [x] bert+cnn
- [x] bert+lstm
- [x] bert+gru
- [x] xlnet
- [ ] xlnet+cnn
- [ ] xlnet+lstm
- [ ] xlnet+gru



# Content

- dataset：存放数据集
- pretrained_models：存放预训练模型
- transformers：transformers文件夹
- results：存放训练结果



# Usage

## 1. 使用不同模型

**在shell文件中修改`model_type`参数即可指定模型**

如，BERT后接FC全连接层，则直接设置`model_type=bert`；BERT后接CNN卷积层，则设置`model_type=bert_cnn`. 

在本README的`Support`中列出了本项目中各个预训练模型支持的`model_type`。

最后，在终端直接运行shell文件即可，如：

```
bash run_classifier.sh
```

## 2. 使用自定义数据集

1. 在`dataset`文件夹里存放自定义的数据集文件夹，如`TestData`.
2. 在根目录下的`utils.py`中，仿照`class THUNewsProcessor`写一个自己的类，如命名为`class TestDataProcessor`，并在`tasks_num_labels`, `processors`, `output_modes`三个dict中添加相应内容.
3. 最后，在你需要运行的shell文件中修改TASK_NAME为你的任务名称，如`TestData`.



# Performance

数据集: THUNews/5_5000

epoch:1

train_steps: 5000 

| model_type     | dev set best F1 and Acc    | remark                                         |
| -------------- | -------------------------- | ---------------------------------------------- |
| bert_base      | 0.9308869881728941, 0.9324 | BERT接FC层, batch_size 8, learning_rate 2e-5   |
| bert_base+cnn  | 0.9136314735833212, 0.9156 | BERT接CNN层, batch_size 8, learning_rate 2e-5  |
| bert_base+lstm | 0.9369254464106703, 0.9372 | BERT接LSTM层, batch_size 8, learning_rate 2e-5 |
| bert_base+gru  | 0.9379539112313108, 0.938  | BERT接GRU层, batch_size 8, learning_rate 2e-5  |
| xlnet_large    | 0.9530066512880131, 0.954  | XLNet接FC层, batch_size 2, learning_rate 2e-5  |
| albert         |                            |                                                |



# Download Chinese Pre-trained Models

[NPL_PEMDC](https://github.com/zhanlaoban/NLP_PEMDC)




