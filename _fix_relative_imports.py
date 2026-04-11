import os

# Fix relative imports in utils modules
files_to_fix = [
    ('src/utils/sting_detection.py', 'from audio_utils import', 'from .audio_utils import'),
    ('src/utils/watermark_detection.py', 'from audio_utils import', 'from .audio_utils import'),
    ('src/utils/watermark_detection_canal.py', 'from audio_utils import', 'from .audio_utils import'),
    ('src/utils/watermark_detection_canal.py', 'from watermark_detection import', 'from .watermark_detection import'),
    ('src/utils/canal_watermark.py', 'from audio_utils import', 'from .audio_utils import'),
]

base_dir = 'C:/Users/raisi/Scripts/motogp_muxing'

for filepath, old, new in files_to_fix:
    full_path = os.path.join(base_dir, filepath)
    if not os.path.exists(full_path):
        print(f'SKIPPING (not found): {filepath}')
        continue

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if old in content:
        content = content.replace(old, new)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'FIXED: {filepath}')
    else:
        print(f'NO CHANGE NEEDED: {filepath}')

print('\nDone fixing relative imports!')
