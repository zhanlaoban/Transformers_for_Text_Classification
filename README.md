# Transformers_for_Text_Classification

# 基于Transformers的文本分类

基于最新的 [huggingface](https://github.com/huggingface) 出品的 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) v2.2.2代码进行重构。为了保证代码日后可以直接复现而不出现兼容性问题，这里将 [transformers](https://github.com/huggingface/transformers/releases/tag/v2.2.2) 放在本地进行调用。



## Content

- dataset：存放数据集
- pretrained_models：存放预训练模型
- transformers：transformers文件夹
- results：存放训练结果



# TODO

- [x] bert+fc
- [ ] bert+cnn
- [ ] bert+lstm
- [ ] bert+gru



# Usage

按照THUNews数据集的例子修改processor即可。