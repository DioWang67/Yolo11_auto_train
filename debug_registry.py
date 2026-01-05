from picture_tool.main_pipeline import build_task_registry, load_config

cfg = load_config()
registry = build_task_registry(cfg)

print("Registered Tasks:")
for name in sorted(registry.keys()):
    print(f" - {name}")
