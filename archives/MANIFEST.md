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

## Agent evaluation v2 (prompt system_v2, 2026-10-14)

| archive | sha256 |
|---|---|
| `main_model_C.tar.gz` | `49f34e8915da446567ae36a66ce4dceaff2deb56dc5159997beabf7fd6683583` |
| `main_toy.tar.gz` | `c5921a08bab1242b01e0490eedc7a0accd0b26b45d13c0feaa97706d93df72cc` |
| `pressure_model_C.tar.gz` | `9703f1ebd6cd70000bf23b3873385b11d3b78c5611fbe34562836c8b195cfd1c` |
| `pressure_toy.tar.gz` | `188623e1a3c0e96d61d6bbde6394a901b0c0e57a9562b104db41773d68a5e60d` |

## Agent evaluation extension (18 new infeasible tasks, 2026-10-15)

Named `<version>_<prompt>_<set>`; each holds `<version>/<prompt>/<set>/` from `runs/agent_eval_ext`. The interrupted pilot starts in `runs/agent_eval_ext/_interrupted` are not archived and not used in any result.

| archive | sha256 |
|---|---|
| `reference_model_C.tar.gz` | `9f2614c63ae7fe03e8ccea938cd1ef45b9592a465884d022f1391a1e15e2e464` |
| `reference_toy.tar.gz` | `a4ced1c3b04ae914d71ce84b22d70c29cd5ba7c76d9da902324b3e4a8d954ab4` |
| `v1_main_model_C.tar.gz` | `94e4949ac90ecd674ecd211e81ef4de10cf7847fb0bb563f1756bed0925feb6f` |
| `v1_main_toy.tar.gz` | `6bb0f6f7fa1c6ac00161a3ec9b98fbd603058a10b1849fa3e5e2edfd6cc688b5` |
| `v1_pressure_model_C.tar.gz` | `3b4e87aac81d0d721cf750f415f6a48f7e922013b63bb7e6cccf12272b555159` |
| `v1_pressure_toy.tar.gz` | `ac94d31a1cc26cf156920f306c37ae4135093b25996565e23c6f6e9433bb54d0` |
| `v2_main_model_C.tar.gz` | `c2d34778e7fbfc58cef5b17b70478d7223becdbfa82416b33fd77c0003c563a0` |
| `v2_main_toy.tar.gz` | `8c27a2aaf5cb3d109fab92616eea652fe04a5a52b5f873367b9a9414a374a986` |
| `v2_pressure_model_C.tar.gz` | `0711bd2e723114439b96999b70bcf962e9e44b89f4a1e5bdc738974753f4b6e3` |
| `v2_pressure_toy.tar.gz` | `d5e6c7bf936ff072aabaf4fdd6cd8d73dda456230a21f85bd76732ad46c13a93` |
