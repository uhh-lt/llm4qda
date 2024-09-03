#!/bin/bash

python document-classification/single-label/single_label_document_classification.py tagesschau gemma2 19291 | tee -a _logs/experiments.gemma2.log
python document-classification/single-label/single_label_document_classification.py bbc gemma2 19291| tee -a _logs/experiments.gemma2.log
python document-classification/single-label/single_label_document_classification.py imdb gemma2 19291| tee -a _logs/experiments.gemma2.log
python document-classification/multi-label/multi_label_document_classification.py imdb gemma2 19291| tee -a _logs/experiments.gemma2.log
python document-classification/hierarchical/hierarchical_document_classification.py tagesschau gemma2 19291| tee -a _logs/experiments.gemma2.log
python document-classification/hierarchical/hierarchical_document_classification.py bbc gemma2 19291| tee -a _logs/experiments.gemma2.log

python extractive-qa/extractive_qa.py squad gemma2 19291| tee -a _logs/experiments.gemma2.log
python extractive-qa/extractive_qa.py squad2 gemma2 19291| tee -a _logs/experiments.gemma2.log
python extractive-qa/extractive_qa.py germanquad gemma2 19291| tee -a _logs/experiments.gemma2.log

python paraphrasing/paraphrasing.py disflqa gemma2 19291| tee -a _logs/experiments.gemma2.log
python paraphrasing/paraphrasing.py disco-en gemma2 19291| tee -a _logs/experiments.gemma2.log
python paraphrasing/paraphrasing.py disco-de gemma2 19291| tee -a _logs/experiments.gemma2.log
python paraphrasing/paraphrasing.py cnndm gemma2 19291| tee -a _logs/experiments.gemma2.log
python paraphrasing/paraphrasing.py mlsum gemma2 19291| tee -a _logs/experiments.gemma2.log

python span-classification/span_classification.py fewnerd-coarse gemma2 19291| tee -a _logs/experiments.gemma2.log
python span-classification/span_classification.py fewnerd-fine gemma2 19291| tee -a _logs/experiments.gemma2.log
python span-classification/span_classification.py germanler-coarse gemma2 19291| tee -a _logs/experiments.gemma2.log
python span-classification/span_classification.py germanler-fine gemma2 19291| tee -a _logs/experiments.gemma2.log
python span-classification/span_classification.py direct-quotation gemma2 19291| tee -a _logs/experiments.gemma2.log

python template-filling/template_filling.py muc4 gemma2 19291| tee -a _logs/experiments.gemma2.log