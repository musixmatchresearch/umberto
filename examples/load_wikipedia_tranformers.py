#!/usr/bin/env python3 -u
# Copyright (c) Musixmatch, spa
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import pipeline

# Fill mask pipeline, with umberto-wikipedia-uncased
fill_mask = pipeline(
	"fill-mask",
	model="Musixmatch/umberto-wikipedia-uncased-v1",
	tokenizer="Musixmatch/umberto-wikipedia-uncased-v1"
)

result = fill_mask("Umberto Eco è <mask> un grande scrittore")
print(result)
