#!/bin/bash

python document-classification/single-label/single_label_document_classification.py tagesschau llama3 | tee -a _logs/experiments.llama3.log
python document-classification/single-label/single_label_document_classification.py bbc llama3 | tee -a _logs/experiments.llama3.log
python document-classification/single-label/single_label_document_classification.py imdb llama3 | tee -a _logs/experiments.llama3.log
python document-classification/multi-label/multi_label_document_classification.py imdb llama3 | tee -a _logs/experiments.llama3.log
python document-classification/hierarchical/hierarchical_document_classification.py tagesschau llama3 | tee -a _logs/experiments.llama3.log
python document-classification/hierarchical/hierarchical_document_classification.py bbc llama3 | tee -a _logs/experiments.llama3.log

python extractive-qa/extractive_qa.py squad llama3 | tee -a _logs/experiments.llama3.log
python extractive-qa/extractive_qa.py squad2 llama3 | tee -a _logs/experiments.llama3.log
python extractive-qa/extractive_qa.py germanquad llama3 | tee -a _logs/experiments.llama3.log

python paraphrasing/paraphrasing.py disflqa llama3 | tee -a _logs/experiments.llama3.log
python paraphrasing/paraphrasing.py disco-en llama3 | tee -a _logs/experiments.llama3.log
python paraphrasing/paraphrasing.py disco-de llama3 | tee -a _logs/experiments.llama3.log
python paraphrasing/paraphrasing.py cnndm llama3 | tee -a _logs/experiments.llama3.log
python paraphrasing/paraphrasing.py mlsum llama3 | tee -a _logs/experiments.llama3.log

python span-classification/span_classification.py fewnerd-coarse llama3 | tee -a _logs/experiments.llama3.log
python span-classification/span_classification.py fewnerd-fine llama3 | tee -a _logs/experiments.llama3.log
python span-classification/span_classification.py germanler-coarse llama3 | tee -a _logs/experiments.llama3.log
python span-classification/span_classification.py germanler-fine llama3 | tee -a _logs/experiments.llama3.log
python span-classification/span_classification.py direct-quotation llama3 | tee -a _logs/experiments.llama3.log

python template-filling/template_filling.py muc4 llama3 | tee -a _logs/experiments.llama3.log