from picture_tool.format import convert_format
from picture_tool.pipeline.core import Task

def run_format_conversion(config, args):
    task_config = config.get("format_conversion", {}).copy()
    # If using wizard defaults, this section might be populated. 
    # If missing, it's safer to not crash, or raise a helpful error if required.
    if not task_config and args.input_format is None: 
         # Assuming if arg is passed, we might not need strict config
         pass 

    if args.input_format:
        task_config["input_formats"] = [args.input_format]
    if args.output_format:
        task_config["output_format"] = args.output_format
    convert_format(task_config)


TASKS = [
    Task(
        name="format_conversion",
        run=run_format_conversion,
        description="Convert image formats in bulk.",
    ),
]
