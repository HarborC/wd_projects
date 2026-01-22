
import os
from pathlib import Path

def save_time(time_dir, process_name, sub_time):
    if isinstance(time_dir, str):
        time_dir = Path(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    minutes, seconds = divmod(sub_time, 60)
    formatted_time = f"{int(minutes)} min {int(seconds)} sec"  
    with open(time_dir / f'train_time.txt', 'a') as f:
        f.write(f'{process_name}: {formatted_time}\n')
