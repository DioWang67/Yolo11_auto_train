"""
Patch script to modify yolo_trainer.py to always include position_config in detection_config.yaml
"""
import re
from pathlib import Path

target_file = Path(r"d:\Git\robotlearning\Yolo11_auto_train\src\picture_tool\train\yolo_trainer.py")

# Read the file
content = target_file.read_text(encoding='utf-8')

# Find and replace the section
old_pattern = r'    if expected_items:\r?\n        payload\["expected_items"\] = expected_items\r?\n    if position_config:\r?\n        payload\["position_config"\] = position_config'

new_code = '''    if expected_items:
        payload["expected_items"] = expected_items
    
    # 始終包含 position_config
    if position_config:
        payload["position_config"] = position_config
    else:
        # 提供預設範本
        payload["position_config"] = {
            "# NOTE": "Position validation disabled. Default template for reference.",
            "enabled": False,
            "ProductA": {
                "Area1": {
                    "tolerance": 10,
                    "expected_boxes": {
                        "ClassName1": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
                    }
                }
            }
        }
        logger.info("Position validation not configured, added default template")'''

# Apply the replacement
new_content = re.sub(old_pattern, new_code, content)

if new_content != content:
    # Backup original
    backup_file = target_file.with_suffix('.py.bak')
    backup_file.write_text(content, encoding='utf-8')
    print(f"Created backup: {backup_file}")
    
    # Write modified content
    target_file.write_text(new_content, encoding='utf-8')
    print(f"✅ Successfully patched {target_file}")
    print("Changed: Always include position_config in detection_config.yaml")
else:
    print("⚠️ Pattern not found or already applied")
