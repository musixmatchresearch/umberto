# Copyright (c) Musixmatch, spa. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


def umberto_wikipedia_uncased(**kwargs):
    from fairseq import hub_utils
    from fairseq.models.roberta.hub_interface import RobertaHubInterface
    x = hub_utils.from_pretrained(
        model_name_or_path='https://mxmdownloads.s3.amazonaws.com/umberto/umberto.wikipedia.uncased.tar.gz',
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='sentencepiece',
        load_checkpoint_heads=True,
        **kwargs,
    )
    return RobertaHubInterface(x['args'], x['task'], x['models'][0])
