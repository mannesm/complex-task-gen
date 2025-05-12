from transformers import TrainerCallback
import os
import json

class CheckpointLoggerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        log_path = os.path.join(args.output_dir, "checkpoints.json")
        checkpoint_info = {
            "global_step": state.global_step,
            "checkpoint_dir": f"checkpoint-{state.global_step}"
        }

        # Append to log file
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(checkpoint_info)

        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
