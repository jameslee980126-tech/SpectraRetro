# Preprocessing file lists for the Table 1 comparisons

Table 1 compares ten methods on the [CHRetro partitions](../../../../datasets/README.md). Four lists identify files in the preprocessing caches used by three of those methods. Each entry gives the path, size and SHA-256 of one file, so readers can check the corresponding preprocessing output.

| Method | Preprocessing file list |
|---|---|
| DiffER | [Full-data preprocessing files](DiffER/full_native_cache.zip.members.json) |
| GLN | [Full-data, class-unknown preprocessing files](GLN/full_unknown_native_cache.zip.members.json) |
| RetroXpert | [Reaction-center files](RetroXpert/full_unknown_reaction_center_cache.zip.members.json) and [sequence-model files](RetroXpert/full_unknown_sequence_cache.zip.members.json), both for the full-data, class-unknown run |

Other methods also convert reaction records into model inputs. For example, Graph2Edits extracts edit actions from mapped reactions, then batches the molecular graphs and edit labels for training. [Graph2Edits edit extraction](../../../../src/baselines/Graph2Edits/source/preprocess.py) · [Graph and label preparation](../../../../src/baselines/Graph2Edits/source/prepare_data.py) · [All ten implementations and their results](../../../../docs/guides/baselines.md)
