'''
this file contains directory paths for each machine it's being run on
define your own data dir here, for e.g.

```
ham10k_dir = "/skin-cancer-mnist-ham10000/versions/2"
```

and then in whatever code that needs to import this dataset:

```
from machine_override import ham10k_dir as data_dir
```

also, to keep this file from being tracked by git
(which I'm doing, along with .gitignore)

```
git update-index --assume-unchanged src/machine_override.py
```

good luck!
'''
