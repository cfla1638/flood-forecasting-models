import time
from rich.progress import track
 
for i in track(range(100),description="进度："):
    time.sleep(0.1)