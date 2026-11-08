"""Create an isolated, allowlisted deployment repository inside digital_twin."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'hosting'
for folder,pattern in [('app','*.py'),('frontend','*'),('artifacts','*.joblib'),('artifacts','*.csv'),('artifacts','workspace.json'),('artifacts','model_registry.json')]:
    for src in (ROOT/folder).glob(pattern):
        if not src.is_file(): continue
        dest=OUT/'digital_twin'/folder/src.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)
for name in ['requirements-lock.txt','DIGITAL_TWIN_GUIDE.md']:
    shutil.copy2(ROOT/name,OUT/name)
print(OUT)
