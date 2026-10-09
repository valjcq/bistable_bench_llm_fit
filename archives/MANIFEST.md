# Archives (pointers and checksums only)

The raw run directories of the agent evaluation (full JSONL logs, raw model responses, tool messages) are not in this repository (about 63 MB for the agent runs; `runs/` is gitignored). Each tarball holds the run directories of one batch, named `<condition>_<set>`. Verify a downloaded tarball with `sha256sum <file>`. They are not published for now.

| archive | sha256 |
|---|---|
| `main_model_C.tar.gz` | `6e9ab329f2e73110f72aabd48b6b2b6d5e9690a8c3429e687b6057144c290321` |
| `main_toy.tar.gz` | `f60d838f012159286a308bc99a73e1dfdca321ffc1a172a037e04f98c1bff8af` |
| `pressure_model_C.tar.gz` | `8f307c92e58258c1cdb95db8b1de419bb96b1fe74838f89cb64b4e924ab77fd3` |
| `pressure_toy.tar.gz` | `f71f6260287a06bb8cf3fc51c4a6bf8319ab7972d31c2291ecb17586ef118dc1` |
| `reference_model_C.tar.gz` | `49a71130b1129ac0259ef0d6da4333478932c8a3d7eab26260f0ad9f36f71c0d` |
| `reference_toy.tar.gz` | `5068c597af2ccbc56475ecbfb45e4f22232b59756a6d2684b216da690dd43410` |
