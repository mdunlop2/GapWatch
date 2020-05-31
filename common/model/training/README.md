Perform model training and evaluation given the simple set of features that were dervived in `basic_featureset_constructor.py`

The features used have the following schema:

| Column  Index | Column Name | Description                                             |
|---------------|-------------|---------------------------------------------------------|
| 0             | label       | Label of the scene, -Danger -No Danger                  |
| 1             | video_url   | Location of the mp4 file that the scene was pulled from |
| 2             | mean_0      | Mean of (i,:,:,0)                                       |
| 3             | mean_1      | Mean of (i,:,:,1)                                       |
| 4             | mean_2      | Mean of (i,:,:,2)                                       |
| 5             | var_0       | Variance of (i,:,:,0)                                   |
| 6             | var_1       | Variance of (i,:,:,1)                                   |
| 7             | var_2       | Variance of (i,:,:,2)                                   |
| 8             | kurt_0      | Kurtosis of (i,:,:,0)                                   |
| 9             | kurt_1      | Kurtosis of (i,:,:,1)                                   |
| 10            | kurt_2      | Kurtosis of (i,:,:,2)                                   |
| 11            | skew_0      | Skewness of (i,:,:,0)                                   |
| 12            | skew_1      | Skewness of (i,:,:,1)                                   |
| 13            | skew_2      | Skewness of (i,:,:,2)                                   |
| 14            | mfcc_0      | Librosa Mel-frequency cepstral coefficient 0            |
| 15            | mfcc_1      | Librosa Mel-frequency cepstral coefficient 1            |
| 16            | mfcc_2      | Librosa Mel-frequency cepstral coefficient 2            |
| 17            | mfcc_3      | Librosa Mel-frequency cepstral coefficient 3            |
| 18            | mfcc_4      | Librosa Mel-frequency cepstral coefficient 4            |
| 19            | mfcc_5      | Librosa Mel-frequency cepstral coefficient 5            |
| 20            | mfcc_6      | Librosa Mel-frequency cepstral coefficient 6            |
| 21            | mfcc_7      | Librosa Mel-frequency cepstral coefficient 7            |
| 22            | mfcc_8      | Librosa Mel-frequency cepstral coefficient 8            |
| 23            | mfcc_9      | Librosa Mel-frequency cepstral coefficient 9            |