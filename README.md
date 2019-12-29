# Transformers_for_Text_Classification

# 基于Transformers的文本分类

基于最新的 [huggingface](https://github.com/huggingface) 出品的 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) v2.2.2代码进行重构。为了保证代码日后可以直接复现而不出现兼容性问题，这里将 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) 放在本地进行调用。



# Content

- dataset：存放数据集
- pretrained_models：存放预训练模型
- transformers：transformers文件夹
- results：存放训练结果



# Support

- [x] bert+fc
- [x] bert+cnn
- [x] bert+lstm
- [x] bert+gru
- [ ] xlnet
- [ ] xlnet+cnn
- [ ] xlnet+lstm
- [ ] xlnet+gru



# Usage

## 1. 使用不同模型

**在shell文件中修改`model_type`参数即可指定模型**

如，BERT后接FC全连接层，则直接设置`model_type=bert`；BERT后接CNN卷积层，则设置`model_type=bert_cnn`. 

在本README的`Performance`的model_type列中附本项目中各个预训练模型支持的`model_type`。

最后，在终端直接运行shell文件即可，如：

```
bash run_classifier.sh
```

## 2. 使用自定义数据集

1. 在`dataset`文件夹里存放自定义的数据集文件夹，如`TestData`.
2. 在根目录下的`utils.py`中，仿照`class THUNewsProcessor`写一个自己的类，如命名为`class TestDataProcessor`，并在`tasks_num_labels`, `processors`, `output_modes`三个dict中添加相应内容.
3. 最后，在你需要运行的shell文件中修改TASK_NAME为你的任务名称，如`TestData`.



# Performance

| model_type | F1   | remark       |
| ---------- | ---- | ------------ |
| bert       |      | BERT接FC层   |
| bert_cnn   |      | BERT接CNN层  |
| bert_lstm  |      | BERT接LSTM层 |
| bert_gru   |      | BERT接GRU层  |
| xlnet      |      |              |
| albert     |      |              |



# Download Chinese Pre-trained Models

[NPL_PEMDC](https://github.com/zhanlaoban/NLP_PEMDC)




