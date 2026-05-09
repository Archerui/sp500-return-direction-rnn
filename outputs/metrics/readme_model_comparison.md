| model               | split      |   accuracy |   macro_f1 |   weighted_f1 |     down_f1 |    flat_f1 |      up_f1 |
|:--------------------|:-----------|-----------:|-----------:|--------------:|------------:|-----------:|-----------:|
| Weighted GRU        | validation |   0.403672 |   0.39351  |      0.405835 |   0.369676  |   0.349338 |   0.461515 |
| Logistic Regression | validation |   0.434333 |   0.32264  |      0.367895 | nan         | nan        | nan        |
| LSTM                | validation |   0.439169 |   0.315079 |      0.357146 |   0.122656  |   0.233312 |   0.589268 |
| RNN                 | validation |   0.445403 |   0.295099 |      0.342    |   0.0773887 |   0.205907 |   0.602001 |
| Random Forest       | validation |   0.444958 |   0.293883 |      0.344021 | nan         | nan        | nan        |
| Majority Class      | validation |   0.446325 |   0.205728 |      0.275465 | nan         | nan        | nan        |