# Span Classification

aka. Auto Coding in DATS

## Datasets

- Fewnerd (https://github.com/thunlp/Few-NERD, https://huggingface.co/datasets/DFKI-SLT/few-nerd)
- German-LER (https://huggingface.co/datasets/elenanereiss/german-ler)
- German Quotations (https://github.com/uhh-lt/german-news-quotation-attribution-2024, https://aclanthology.org/2024.lrec-main.394.pdf)
- German NER? (https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germaner.html)

## Results

- [x] Fewnerd - Coarse

```
❯ python span_classification.py fewnerd-coarse
Using the following labels:
1. Art
2. Building
3. Event
4. Location
5. Organization
6. Other
7. Person
8. Product

Using the following examples:
Art: Mona Lisa
Building: Eiffel Tower
Evaluating: 10000it [1:38:32,  1.69it/s]
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
F1-Score:  0.49572288520435104
Accuracy:  0.8314513199939602
              precision    recall  f1-score   support

       erson       0.69      0.79      0.74      5713
     ocation       0.61      0.59      0.60      7138
 rganization       0.47      0.59      0.52      5144
      roduct       0.23      0.47      0.31      1657
          rt       0.38      0.46      0.42      1143
        ther       0.17      0.08      0.11      2552
     uilding       0.32      0.30      0.31      1267
        vent       0.11      0.42      0.18      1107

   micro avg       0.45      0.55      0.50     25721
   macro avg       0.37      0.46      0.40     25721
weighted avg       0.48      0.55      0.51     25721
```

- [ ] Fewnerd - Fine

```
❯ python span_classification.py fewnerd-fine
Using the following labels:
1. Art - Broadcastprogram
2. Art - Film
3. Art - Music
4. Art - Other
5. Art - Painting
6. Art - Writtenart
7. Building - Airport
8. Building - Hospital
9. Building - Hotel
10. Building - Library
11. Building - Other
12. Building - Restaurant
13. Building - Sportsfacility
14. Building - Theater
15. Event - Attack/Battle/War/Militaryconflict
16. Event - Disaster
17. Event - Election
18. Event - Other
19. Event - Protest
20. Event - Sportsevent
21. Location - Gpe
22. Location - Bodiesofwater
23. Location - Island
24. Location - Mountain
25. Location - Other
26. Location - Park
27. Location - Road/Railway/Highway/Transit
28. Organization - Company
29. Organization - Education
30. Organization - Government/Governmentagency
31. Organization - Media/Newspaper
32. Organization - Other
33. Organization - Politicalparty
34. Organization - Religion
35. Organization - Showorganization
36. Organization - Sportsleague
37. Organization - Sportsteam
38. Other - Astronomything
39. Other - Award
40. Other - Biologything
41. Other - Chemicalthing
42. Other - Currency
43. Other - Disease
44. Other - Educationaldegree
45. Other - God
46. Other - Language
47. Other - Law
48. Other - Livingthing
49. Other - Medical
50. Person - Actor
51. Person - Artist/Author
52. Person - Athlete
53. Person - Director
54. Person - Other
55. Person - Politician
56. Person - Scholar
57. Person - Soldier
58. Product - Airplane
59. Product - Car
60. Product - Food
61. Product - Game
62. Product - Other
63. Product - Ship
64. Product - Software
65. Product - Train
66. Product - Weapon

Using the following examples:
Art - Painting: Mona Lisa
Building - Other: Eiffel Tower
Evaluating: 10000it [1:54:36,  1.45it/s]
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - athlete seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - sportsevent seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - GPE seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - disease seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - company seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - attack/battle/war/militaryconflict seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - award seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - law seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - artist/author seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - showorganization seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - politician seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - park seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - livingthing seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - actor seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - politicalparty seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - education seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - road/railway/highway/transit seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - government/governmentagency seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - soldier seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - sportsfacility seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - sportsteam seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - car seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - theater seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - astronomything seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - scholar seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - weapon seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - broadcastprogram seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - writtenart seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - hotel seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - disaster seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - biologything seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - library seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - music seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - mountain seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - game seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - airplane seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - software seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - film seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - restaurant seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - protest seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - other seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - medical seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - media/newspaper seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - bodiesofwater seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - ship seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - language seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: location - island seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - chemicalthing seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - currency seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person - director seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - religion seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - hospital seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - food seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organization - sportsleague seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: product - train seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - educationaldegree seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: other - god seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: event - election seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: building - airport seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: art - painting seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
F1-Score:  0.39314857886230137
Accuracy:  0.8233668926170936
                                     precision    recall  f1-score   support

                                GPE       0.67      0.49      0.57      5340
                              actor       0.77      0.60      0.68       416
                           airplane       0.43      0.49      0.46       227
                            airport       0.55      0.72      0.62       106
                      artist/author       0.66      0.19      0.29       950
                     astronomything       0.37      0.54      0.44       219
                            athlete       0.77      0.54      0.64       775
 attack/battle/war/militaryconflict       0.31      0.57      0.40       300
                              award       0.39      0.42      0.40       260
                       biologything       0.27      0.47      0.34       461
                      bodiesofwater       0.46      0.52      0.49       288
                   broadcastprogram       0.11      0.55      0.19       167
                                car       0.48      0.49      0.48       177
                      chemicalthing       0.45      0.36      0.40       238
                            company       0.45      0.51      0.48      1038
                           currency       0.45      0.50      0.47       214
                           director       0.72      0.44      0.54       154
                           disaster       0.24      0.37      0.29        54
                            disease       0.31      0.55      0.40       254
                          education       0.61      0.58      0.59       559
                  educationaldegree       0.23      0.39      0.29        97
                           election       0.06      0.20      0.10        44
                               film       0.35      0.56      0.43       247
                               food       0.19      0.12      0.15        88
                               game       0.36      0.27      0.31       146
                                god       0.73      0.32      0.45       180
        government/governmentagency       0.16      0.38      0.23       394
                           hospital       0.61      0.56      0.59        96
                              hotel       0.43      0.45      0.44        67
                             island       0.60      0.49      0.54       173
                           language       0.62      0.42      0.50       200
                                law       0.32      0.26      0.28       139
                            library       0.56      0.52      0.54       107
                        livingthing       0.07      0.01      0.01       177
                    media/newspaper       0.49      0.25      0.33       378
                            medical       0.04      0.06      0.05       115
                           mountain       0.69      0.46      0.55       191
                              music       0.14      0.36      0.20       276
                              other       0.29      0.29      0.29      5506
                           painting       0.10      0.38      0.16        16
                               park       0.54      0.32      0.40       101
                     politicalparty       0.42      0.48      0.45       306
                         politician       0.60      0.26      0.36       762
                            protest       0.09      0.04      0.06        49
                           religion       0.27      0.25      0.26       188
                         restaurant       0.14      0.09      0.11        44
       road/railway/highway/transit       0.59      0.43      0.50       491
                            scholar       0.34      0.07      0.12       187
                               ship       0.23      0.19      0.21       108
                   showorganization       0.09      0.08      0.08       181
                           software       0.32      0.20      0.24       260
                            soldier       0.18      0.09      0.12       158
                        sportsevent       0.19      0.23      0.21       435
                     sportsfacility       0.49      0.37      0.42       107
                       sportsleague       0.35      0.27      0.31       267
                         sportsteam       0.69      0.40      0.50       644
                            theater       0.53      0.39      0.45       102
                              train       0.16      0.27      0.20        66
                             weapon       0.32      0.16      0.21       162
                         writtenart       0.14      0.45      0.22       287

                          micro avg       0.40      0.39      0.39     25739
                          macro avg       0.39      0.36      0.35     25739
                       weighted avg       0.46      0.39      0.41     25739

```

- [ ] German-LER - Coarse

```
❯ python span_classification.py germanler-coarse
Using the following labels:
1. Person
2. Ort
3. Organisation
4. Norm
5. Gesetz
6. Rechtsprechung
7. Literatur

Using the following examples:
Person: Angela Merkel
Gesetz: Artikel 5
Evaluating: 6673it [1:26:54,  1.28it/s]
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: organisation seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: norm seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: gesetz seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: person seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: rechtsprechung seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: literatur seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: ort seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
F1-Score:  0.2540634809905825
Accuracy:  0.8613263950398583
               precision    recall  f1-score   support

echtsprechung       0.19      0.13      0.16      1244
        erson       0.10      0.63      0.17       324
        esetz       0.02      0.04      0.03       354
     iteratur       0.07      0.10      0.08       314
          orm       0.38      0.48      0.42      2037
  rganisation       0.15      0.38      0.21       795
           rt       0.29      0.50      0.37       250

    micro avg       0.20      0.34      0.25      5318
    macro avg       0.17      0.32      0.21      5318
 weighted avg       0.24      0.34      0.27      5318

```

- [x] German-LER - Fine

```
❯ python span_classification.py germanler-fine
Using the following labels:
1. Person
2. Anwalt
3. Richter
4. Land
5. Stadt
6. Straße
7. Landschaft
8. Organisation
9. Unternehmen
10. Institution
11. Gericht
12. Marke
13. Gesetz
14. Verordnung
15. Eu Norm
16. Vorschrift
17. Vertrag
18. Gerichtsentscheidung
19. Literatur

Using the following examples:
Person: Angela Merkel
Gesetz: Artikel 5
Evaluating: 6673it [1:56:57,  1.05s/it]
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Institution seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Gesetz seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Vorschrift seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Person seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Unternehmen seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Gerichtsentscheidung seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Gericht seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Richter seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: EU Norm seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Literatur seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Land seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Vertrag seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Anwalt seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Organisation seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Marke seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Verordnung seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Stadt seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Landschaft seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
/home/tfischer/micromamba/envs/dwts/lib/python3.11/site-packages/seqeval/metrics/sequence_labeling.py:171: UserWarning: Straße seems not to be NE tag.
  warnings.warn('{} seems not to be NE tag.'.format(chunk))
F1-Score:  0.20589721988205564
Accuracy:  0.8343021110126956
                     precision    recall  f1-score   support

             U Norm       0.68      0.03      0.06      1408
                and       0.48      0.70      0.57       149
          andschaft       0.00      0.00      0.00        22
               arke       0.08      0.44      0.14        32
             ericht       0.09      0.20      0.13       321
erichtsentscheidung       0.06      0.02      0.03      1244
          erordnung       0.05      0.04      0.05        71
              erson       0.08      0.54      0.14       173
             ertrag       0.06      0.02      0.03       290
              esetz       0.54      0.41      0.47      1816
             ichter       0.10      0.03      0.04       142
           iteratur       0.07      0.11      0.09       314
         nstitution       0.02      0.02      0.02       222
         nternehmen       0.32      0.21      0.25       108
              nwalt       0.00      0.00      0.00         9
          orschrift       0.01      0.02      0.01        64
               tadt       0.33      0.54      0.41        67
              traße       0.50      0.76      0.60        21

          micro avg       0.23      0.19      0.21      6473
          macro avg       0.19      0.23      0.17      6473
       weighted avg       0.35      0.19      0.19      6473

```

- [x] German Quotations
      Only Speaker and Direct speech, only the labels, no description

```
Using the following labels:
1. Sprecher
2. Direkte Rede

Using the following examples:
Sprecher: Angela Merkel
Direkte Rede: "Wir schaffen das!"
Evaluating: 434it [11:44,  1.62s/it]
F1-Score:  0.29628422425032597
Accuracy:  0.8973878588194758
              precision    recall  f1-score   support

 irekte Rede       0.65      0.33      0.44       853
     precher       0.29      0.23      0.26      2712

   micro avg       0.35      0.25      0.30      3565
   macro avg       0.47      0.28      0.35      3565
weighted avg       0.38      0.25      0.30      3565

```
