import json
import os
import glob

def sanitize_notebooks():
    for nb in glob.glob('src/notebooks/*.ipynb'):
        print(f"Checking {nb}...")
        try:
            with open(nb, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            meta = data.get('metadata', {})
            if 'widgets' in meta:
                del meta['widgets']
                print(f"  - Removed widgets metadata")
                modified = True
            
            content_str = json.dumps(data)
            secret = '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'
            if secret in content_str:
                content_str = content_str.replace(secret, 'hf_REDACTED')
                data = json.loads(content_str)
                print(f"  - Redacted token")
                modified = True
            
            if modified:
                with open(nb, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=1)
                print(f"  - Saved changes to {nb}")
        except Exception as e:
            print(f"  - Error processing {nb}: {e}")

if __name__ == "__main__":
    sanitize_notebooks()
