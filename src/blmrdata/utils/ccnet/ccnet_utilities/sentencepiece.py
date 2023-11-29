# -*- coding: utf-8 -*-
""" SentencePiece and KenLM models for tokenization and perplexity estimation

from repository: https://github.com/facebookresearch/cc_net/cc_net/perplexity.py

@Author: Evan Dufraisse
@Date: Thu Nov 30 2023
@Contact: e[dot]dufraisse[at]gmail[dot]com
@License: MIT License
"""

import sentencepiece
from pathlib import Path
from blmrdata.utils.ccnet.ccnet_utilities import text_normalizer

class SentencePiece:
    """
    Class to tokenize a text with SentencePiece
    """
    # Sentence Pieces model have to be read back from disk.
    warning_when_pickling = True

    def __init__(
        self,
        model: Path,
        field: str = "text",
        output_field: str = "tokenized",
        normalize: bool = True,
        create_upon_init: bool = True,
    ):
        super().__init__()
        self.model = model
        self.field = field
        self.output_field = output_field
        self.normalize = normalize
        self.sp: sentencepiece.SentencePieceProcessor = None
        if create_upon_init:
            self._prepare()

    def _prepare(self):
        if self.sp is not None:
            return
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(self.model))
        return self

    def do(self, document: dict) -> dict:
        """Tokenize a document dict entry"""
        text = document[self.field]
        if self.normalize:
            text = text_normalizer.normalize(text)
        tokenized = self.sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document
    

    
    def __call__(self, text: str) -> dict:
        """Takes a string and returns a dict with the orginal and the tokenized texts"""
        return self.do({self.field: text})
        
