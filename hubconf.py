dependencies = [
    'fairseq',
    'sentencepiece',
    'torch',
]

def umberto_commoncrawl_cased(**kwargs):
    from fairseq import hub_utils
    from fairseq.models.roberta.hub_interface import RobertaHubInterface
    x = hub_utils.from_pretrained(
        model_name_or_path='https://mxmdownloads.s3.amazonaws.com/umberto/umberto.commoncrawl.cased.tar.gz',
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='sentencepiece',
        load_checkpoint_heads=True,
        **kwargs,
    )
    return RobertaHubInterface(x['args'], x['task'], x['models'][0])
