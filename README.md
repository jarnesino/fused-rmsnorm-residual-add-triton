# Fused RMSNorm and Residual Add in Triton

A Triton GPU kernel project that fuses the residual-add and RMSNorm steps used at every transformer layer boundary into a single memory pass.

## Setup

1. Install Docker Desktop or restart it if it is already installed
2. Add execution permissions to the pre-commit hook with `chmod +x .github/hooks/*`.
3. Add the version-controlled hooks directory as a hooks path for the local git command with `git config core.hooksPath .github/hooks`.
4. Install Task: `brew install go-task`
5. Build the image and set up the requirements: `task setup` (first run takes a while)
6. Check that the smoke test runs green: `task test` 
7. Check that you can Bash inside the container: `task shell`

## Commands

| Command             | Where it runs     | What it does                                                                              |
|---------------------|-------------------|-------------------------------------------------------------------------------------------|
| `task`              | Local             | List all tasks                                                                            |
| `task setup`        | Local + container | First-time install (build image, resolve lock, install CPU dependencies)                  |
| `task build`        | Local             | Rebuild the Docker image (only needed after editing the Dockerfile)                       |
| `task lock`         | Container         | Resolve dependency versions into `uv.lock`                                                |
| `task sync`         | Container         | Install dependencies from `uv.lock` into the virtual environment volume (CPU torch)       |
| `task add -- <pkg>` | Container         | Add a dependency to `pyproject.toml`, then re-lock and re-sync                            |
| `task shell`        | Container         | Open an interactive bash session                                                          |
| `task py`           | Container         | Open a Python REPL with the project installed                                             |
| `task lint`         | Container         | Run ruff check and ruff format in check mode, reports problems without changing files     |
| `task fmt`          | Container         | Run ruff with autofix and reformat files in place                                         |
| `task test`         | Container         | Run the CPU test suite under `TRITON_INTERPRET=1` (same command as CI)                    |
| `task check`        | Container         | Run lint then test (same command as the pre-commit hook)                                  |
| `task gpu:sync`     | GPU host          | Installs CUDA torch plus bench extras. Run natively on Colab or a GPU box, not via Docker |
| `task gpu:test`     | GPU host          | Run the full test matrix including gpu-marked tests                                       |
| `task gpu:bench`    | GPU host          | Run the benchmark                                                                         |
| `task clean`        | Local             | Remove containers and venv/cache volumes (full reset)                                     |

## Notes

- When simulating, keep test shapes small. At most 16 x 4096 (token count M <= 16, hidden size N <= 4096). The simulation interpreter runs one Python iteration per row, so a large token count will make tests slow.