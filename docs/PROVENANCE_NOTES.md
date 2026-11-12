# Paths

The tree root is `/data`.

- Master tables: `/data/data_split/raw/`
- Splits and wild-type PDBs: `/data/data_split/splits/`
- ESM-FNN code and weight file: `/data/ESM-FNN/`
- MMseqs2: `/data/tools/mmseqs/bin/mmseqs`
- US-align: `/data/tools/USalign`
- FoldX: `/data/tools/foldx`

`ESM-FNN/protocol.py` reads those locations through `set_roots`. The default root is `/data`.
