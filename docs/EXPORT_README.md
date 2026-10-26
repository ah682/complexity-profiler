# JSON and CSV exports

The [export guide](EXPORT_GUIDE.md) contains a complete profiling example, API
reference, field descriptions and error handling.

For a quick command-line export:

```bash
complexity-profiler analyze merge_sort --sizes 100,500,1000,5000 --runs 5 --no-visualize --export json --output results/merge-sort.json
complexity-profiler analyze merge_sort --sizes 100,500,1000,5000 --runs 5 --no-visualize --export csv --output results/merge-sort.csv
```

JSON keeps individual runs and fit scores. CSV provides one aggregate row per
input size. The two commands above create separate profiles; use the Python API
in the guide when both files should describe the same run.
