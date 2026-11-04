The tree has two relation occurrences, `R0` and `R1`, joined by equality on `edge_0_key`.

| Relation | Tuple occurrence ID | Join key | Weight |
| --- | ---: | ---: | ---: |
| R0 | 101 | 7 | 0.8 |
| R0 | 102 | 7 | 0.25 |
| R0 | 103 | 8 | 0 |
| R1 | 201 | 7 | 1 |
| R1 | 202 | 7 | 0.5 |
| R1 | 203 | 8 | 1 |

The complete join has five tuple-ID vectors. Each call independently includes each vector with the product of its row weights.

| Complete tuple-ID vector | Inclusion probability | Core branch |
| --- | ---: | --- |
| `[101, 201]` | 0.8 | heavy |
| `[101, 202]` | 0.4 | light |
| `[102, 201]` | 0.25 | light |
| `[102, 202]` | 0.125 | light |
| `[103, 203]` | 0 | zero mass |

The first two vectors share row 101, and the first and third share row 201. Their inclusion events are independent. Expected sample size is `mu = 0.8 + 0.4 + 0.25 + 0.125 = 63/40 = 1.575`. Each realized subset has between zero and four elements.

Run from the artifact directory after building:

```sh
./build/product_sample --query-dir examples/tiny --weights weights --root 0 --seed 42 --calls 5
```

Each JSON line gives the zero-based call number, `mu`, and `sample`. Each member of `sample` lists one tuple occurrence ID per relation in numeric relation order. The CLI orders the members lexicographically. The root can be 0 or 1, and controls the orientation used to build the index.

`structure.tsv` uses the existing seven-field relation and edge records. The relation feature-file field is `unused` because sampling reads key and weight files. The `verification_root` field belongs to the preparation format. The CLI selects its root through `--root` and uses the keys and weights directly.
