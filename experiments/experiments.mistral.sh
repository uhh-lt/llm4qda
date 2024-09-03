#!/bin/bash

python document-classification/single-label/single_label_document_classification.py tagesschau mistral 19290 | tee -a _logs/experiments.mistral.log
python document-classification/single-label/single_label_document_classification.py bbc mistral 19290| tee -a _logs/experiments.mistral.log
python document-classification/single-label/single_label_document_classification.py imdb mistral 19290| tee -a _logs/experiments.mistral.log
python document-classification/multi-label/multi_label_document_classification.py imdb mistral 19290| tee -a _logs/experiments.mistral.log
python document-classification/hierarchical/hierarchical_document_classification.py tagesschau mistral 19290| tee -a _logs/experiments.mistral.log
python document-classification/hierarchical/hierarchical_document_classification.py bbc mistral 19290| tee -a _logs/experiments.mistral.log

python extractive-qa/extractive_qa.py squad mistral 19290| tee -a _logs/experiments.mistral.log
python extractive-qa/extractive_qa.py squad2 mistral 19290| tee -a _logs/experiments.mistral.log
python extractive-qa/extractive_qa.py germanquad mistral 19290| tee -a _logs/experiments.mistral.log

python paraphrasing/paraphrasing.py disflqa mistral 19290| tee -a _logs/experiments.mistral.log
python paraphrasing/paraphrasing.py disco-en mistral 19290| tee -a _logs/experiments.mistral.log
python paraphrasing/paraphrasing.py disco-de mistral 19290| tee -a _logs/experiments.mistral.log
python paraphrasing/paraphrasing.py cnndm mistral 19290| tee -a _logs/experiments.mistral.log
python paraphrasing/paraphrasing.py mlsum mistral 19290| tee -a _logs/experiments.mistral.log

python span-classification/span_classification.py fewnerd-coarse mistral 19290| tee -a _logs/experiments.mistral.log
python span-classification/span_classification.py fewnerd-fine mistral 19290| tee -a _logs/experiments.mistral.log
python span-classification/span_classification.py germanler-coarse mistral 19290| tee -a _logs/experiments.mistral.log
python span-classification/span_classification.py germanler-fine mistral 19290| tee -a _logs/experiments.mistral.log
python span-classification/span_classification.py direct-quotation mistral 19290| tee -a _logs/experiments.mistral.log

python template-filling/template_filling.py muc4 mistral 19290| tee -a _logs/experiments.mistral.log