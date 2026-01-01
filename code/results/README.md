# Results directory

This directory is populated by running the reproducibility pipeline:

```bash
cd ..
python run_all.py
```

Each experiment writes a CSV or JSON file here. These files back every
number reported in the paper. The directory is empty at submission time;
run the pipeline to regenerate all outputs.

See `../REPRODUCE.md` for full instructions.
