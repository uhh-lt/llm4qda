# Extractive Question Answering

## Datasets

- SQUAD
- SQUADv2: https://huggingface.co/datasets/rajpurkar/squad_v2
- GermanQUAD: https://huggingface.co/datasets/deepset/germanquad, https://www.deepset.ai/germanquad

## Idea
We want to support as many use cases as possible, and therefore need a very generic approach.

```
Metadata - Typ - Frage

Domain: News
Veröffentlichungsdatum  - DATE  - Wann wurde das Dokument veröffentlicht?
Autor                   - STR   - Wer hat das Dokument verfasst?
Publisher               - STR   - Unter welchem Unternehmen hat das Dokument veröffentlicht?

Domain: Interviews
Alter                   - INT   - Wie alt ist die Interviewte Person?
Interviewdatum          - DATE  - Wann wurde das Interview geführt?

Domain: Political debates
Ausschuss               - STR   - Von welchem Ausschuss war die Rede?
Redner                  - STR   - Wer hat die Rede gehalten? Wer ist der Redner?
Tagesordnungspunkt      - STR   - Zu welchem Tagesordnungspunkt wurde die Rede gehalten?

Domain: Finance
Umsatz                  - INT   - Was ist der Umsatz des Unternehmens?
```

## Related

- https://www.deepset.ai/blog/automating-information-extraction-with-question-answering
- https://arxiv.org/abs/2304.10994
