class PreTrainedModel():
	def __init__(self):
		pass

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
		config = kwargs.pop('config', None)
		print(config)


class BertPreTrainedModel(PreTrainedModel):
	def __init__(self):
		pass


class BertForSequenceClassification(BertPreTrainedModel):
	def __init__(self, config, args):
		print(args)
		print(config)
		pass



model_name_or_path = 'asfafs'
config = 'cvxcvxcv'
args = '123'

BertForSequenceClassification(config=config, args=args).from_pretrained(model_name_or_path, config=config)