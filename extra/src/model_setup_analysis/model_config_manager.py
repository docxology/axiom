"""
Model Configuration Manager for AXIOM Extensions

Manages model configurations, presets, and configuration validation
for AXIOM architecture extensions.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPreset:
    """Represents a model configuration preset."""
    name: str
    model_type: str
    description: str
    config: Dict[str, Any]
    performance_target: str = "balanced"  # "speed", "accuracy", "balanced"
    memory_efficient: bool = False


class ModelConfigManager:
    """
    Manages model configurations and presets for AXIOM extensions.

    This class provides methods to:
    - Load and save model configurations
    - Manage configuration presets
    - Validate configuration compatibility
    - Generate optimized configurations
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory for storing configurations (if None, uses output config)
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if config_dir is None:
                # Use centralized output configuration
                self.config_dir = output_config.get_path_for("models", "configs")
            else:
                self.config_dir = Path(config_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "model_configs"

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._presets: Dict[str, ModelPreset] = {}
        self._load_default_presets()

    def _load_default_presets(self):
        """Load default model configuration presets."""
        default_presets = [
            ModelPreset(
                name="rmm_light",
                model_type="rmm",
                description="Lightweight RMM for quick experimentation",
                config={
                    "num_slots": 5,
                    "slot_size": 32,
                    "hidden_dim": 128,
                    "num_objects": 3,
                    "learning_rate": 0.001
                },
                performance_target="speed",
                memory_efficient=True
            ),
            ModelPreset(
                name="rmm_standard",
                model_type="rmm",
                description="Standard RMM configuration",
                config={
                    "num_slots": 10,
                    "slot_size": 64,
                    "hidden_dim": 256,
                    "num_objects": 5,
                    "learning_rate": 0.001
                },
                performance_target="balanced"
            ),
            ModelPreset(
                name="rmm_heavy",
                model_type="rmm",
                description="High-capacity RMM for complex tasks",
                config={
                    "num_slots": 20,
                    "slot_size": 128,
                    "hidden_dim": 512,
                    "num_objects": 10,
                    "learning_rate": 0.0005
                },
                performance_target="accuracy"
            ),
            ModelPreset(
                name="hmm_efficient",
                model_type="hmm",
                description="Memory-efficient HMM configuration",
                config={
                    "num_layers": 2,
                    "hidden_dim": 128,
                    "memory_size": 64,
                    "attention_heads": 4,
                    "learning_rate": 0.001
                },
                performance_target="speed",
                memory_efficient=True
            ),
            ModelPreset(
                name="hmm_balanced",
                model_type="hmm",
                description="Balanced HMM configuration",
                config={
                    "num_layers": 3,
                    "hidden_dim": 256,
                    "memory_size": 128,
                    "attention_heads": 8,
                    "learning_rate": 0.001
                },
                performance_target="balanced"
            ),
            ModelPreset(
                name="smm_compact",
                model_type="smm",
                description="Compact SMM for resource-constrained environments",
                config={
                    "num_slots": 8,
                    "slot_dim": 64,
                    "hidden_dim": 128,
                    "learning_rate": 0.001
                },
                performance_target="speed",
                memory_efficient=True
            )
        ]

        for preset in default_presets:
            self._presets[preset.name] = preset

    def get_preset(self, name: str) -> Optional[ModelPreset]:
        """
        Get a preset by name.

        Args:
            name: Name of the preset

        Returns:
            ModelPreset if found, None otherwise
        """
        return self._presets.get(name)

    def list_presets(self, model_type: Optional[str] = None) -> List[ModelPreset]:
        """
        List available presets, optionally filtered by model type.

        Args:
            model_type: Filter by model type (optional)

        Returns:
            List of matching presets
        """
        presets = list(self._presets.values())

        if model_type:
            presets = [p for p in presets if p.model_type == model_type]

        return presets

    def create_preset(self, preset: ModelPreset) -> None:
        """
        Create a new preset.

        Args:
            preset: Preset to create
        """
        self._presets[preset.name] = preset
        self.logger.info(f"Created preset: {preset.name}")

    def save_preset(self, preset: ModelPreset, filename: Optional[str] = None) -> None:
        """
        Save a preset to file.

        Args:
            preset: Preset to save
            filename: Optional filename (defaults to preset name)
        """
        if filename is None:
            filename = f"{preset.name}.json"

        filepath = self.config_dir / filename

        data = {
            "name": preset.name,
            "model_type": preset.model_type,
            "description": preset.description,
            "config": preset.config,
            "performance_target": preset.performance_target,
            "memory_efficient": preset.memory_efficient
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved preset to: {filepath}")

    def load_preset(self, filepath: Union[str, Path]) -> ModelPreset:
        """
        Load a preset from file.

        Args:
            filepath: Path to preset file

        Returns:
            Loaded ModelPreset

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Preset file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        preset = ModelPreset(
            name=data["name"],
            model_type=data["model_type"],
            description=data["description"],
            config=data["config"],
            performance_target=data.get("performance_target", "balanced"),
            memory_efficient=data.get("memory_efficient", False)
        )

        self._presets[preset.name] = preset
        self.logger.info(f"Loaded preset: {preset.name}")

        return preset

    def generate_optimized_config(
        self,
        model_type: str,
        target_performance: str = "balanced",
        memory_limit_gb: Optional[float] = None,
        speed_priority: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an optimized configuration based on requirements.

        Args:
            model_type: Type of model (rmm, hmm, imm, smm, tmm)
            target_performance: Target performance level ("speed", "accuracy", "balanced")
            memory_limit_gb: Memory limit in GB (optional)
            speed_priority: Whether to prioritize speed over accuracy

        Returns:
            Optimized configuration dictionary
        """
        base_configs = {
            "rmm": {
                "num_slots": 10,
                "slot_size": 64,
                "hidden_dim": 256,
                "num_objects": 5,
                "learning_rate": 0.001
            },
            "hmm": {
                "num_layers": 3,
                "hidden_dim": 256,
                "memory_size": 128,
                "attention_heads": 8,
                "learning_rate": 0.001
            },
            "imm": {
                "hidden_dim": 256,
                "num_layers": 2,
                "learning_rate": 0.001
            },
            "smm": {
                "num_slots": 8,
                "slot_dim": 64,
                "hidden_dim": 128,
                "learning_rate": 0.001
            },
            "tmm": {
                "hidden_dim": 256,
                "num_layers": 2,
                "temporal_horizon": 10,
                "learning_rate": 0.001
            }
        }

        if model_type not in base_configs:
            raise ValueError(f"Unsupported model type: {model_type}")

        config = base_configs[model_type].copy()

        # Adjust based on performance target
        if target_performance == "speed":
            self._optimize_for_speed(config, model_type)
        elif target_performance == "accuracy":
            self._optimize_for_accuracy(config, model_type)
        elif target_performance == "balanced":
            pass  # Use base config

        # Apply memory constraints
        if memory_limit_gb is not None:
            config = self._apply_memory_constraints(config, model_type, memory_limit_gb)

        # Apply speed priority adjustments
        if speed_priority:
            self._apply_speed_priority(config, model_type)

        return config

    def _optimize_for_speed(self, config: Dict[str, Any], model_type: str) -> None:
        """Optimize configuration for speed."""
        if model_type == "rmm":
            config["num_slots"] = min(config["num_slots"], 5)
            config["slot_size"] = min(config["slot_size"], 32)
            config["hidden_dim"] = min(config["hidden_dim"], 128)
        elif model_type == "hmm":
            config["num_layers"] = min(config["num_layers"], 2)
            config["hidden_dim"] = min(config["hidden_dim"], 128)
            config["attention_heads"] = min(config["attention_heads"], 4)

    def _optimize_for_accuracy(self, config: Dict[str, Any], model_type: str) -> None:
        """Optimize configuration for accuracy."""
        if model_type == "rmm":
            config["num_slots"] = max(config["num_slots"], 15)
            config["slot_size"] = max(config["slot_size"], 96)
            config["hidden_dim"] = max(config["hidden_dim"], 384)
        elif model_type == "hmm":
            config["num_layers"] = max(config["num_layers"], 4)
            config["hidden_dim"] = max(config["hidden_dim"], 384)
            config["attention_heads"] = max(config["attention_heads"], 12)

    def _apply_memory_constraints(self, config: Dict[str, Any], model_type: str, memory_limit_gb: float) -> Dict[str, Any]:
        """Apply memory constraints to configuration."""
        # Rough memory estimation (simplified)
        memory_estimate = self._estimate_memory_usage(config, model_type)

        if memory_estimate > memory_limit_gb:
            # Scale down configuration to fit memory
            scale_factor = memory_limit_gb / memory_estimate

            if model_type == "rmm":
                config["hidden_dim"] = int(config["hidden_dim"] * scale_factor)
                config["slot_size"] = int(config["slot_size"] * scale_factor)
            elif model_type == "hmm":
                config["hidden_dim"] = int(config["hidden_dim"] * scale_factor)
                config["memory_size"] = int(config["memory_size"] * scale_factor)

        return config

    def _apply_speed_priority(self, config: Dict[str, Any], model_type: str) -> None:
        """Apply speed priority adjustments."""
        # Reduce batch processing complexity
        if model_type in ["rmm", "hmm"]:
            config["batch_size"] = config.get("batch_size", 32)
            config["gradient_checkpointing"] = True

    def _estimate_memory_usage(self, config: Dict[str, Any], model_type: str) -> float:
        """Estimate memory usage for a configuration (simplified)."""
        hidden_dim = config.get("hidden_dim", 256)

        if model_type == "rmm":
            num_slots = config.get("num_slots", 10)
            slot_size = config.get("slot_size", 64)
            return (num_slots * slot_size * hidden_dim * 4) / (1024**3)  # float32
        elif model_type == "hmm":
            memory_size = config.get("memory_size", 128)
            num_layers = config.get("num_layers", 3)
            return (memory_size * hidden_dim * num_layers * 4) / (1024**3)
        else:
            return (hidden_dim * 1000 * 4) / (1024**3)  # rough estimate

    def validate_config(self, config: Dict[str, Any], model_type: str) -> List[str]:
        """
        Validate a model configuration.

        Args:
            config: Configuration to validate
            model_type: Type of model

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Required fields validation
        required_fields = {
            "rmm": ["num_slots", "slot_size", "hidden_dim", "num_objects"],
            "hmm": ["num_layers", "hidden_dim", "memory_size", "attention_heads"],
            "imm": ["hidden_dim", "num_layers"],
            "smm": ["num_slots", "slot_dim", "hidden_dim"],
            "tmm": ["hidden_dim", "num_layers", "temporal_horizon"]
        }

        if model_type not in required_fields:
            errors.append(f"Unsupported model type: {model_type}")
            return errors

        for field in required_fields[model_type]:
            if field not in config:
                errors.append(f"Missing required field for {model_type}: {field}")

        # Value validation
        if "hidden_dim" in config and config["hidden_dim"] <= 0:
            errors.append("hidden_dim must be positive")

        if "num_slots" in config and config["num_slots"] <= 0:
            errors.append("num_slots must be positive")

        if "learning_rate" in config and not (0 < config["learning_rate"] <= 1):
            errors.append("learning_rate must be between 0 and 1")

        return errors

    def interpolate_configs(self, config1: Dict[str, Any], config2: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        """
        Interpolate between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration
            ratio: Interpolation ratio (0.0 = config1, 1.0 = config2)

        Returns:
            Interpolated configuration
        """
        if not (0 <= ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1")

        interpolated = {}

        # Get all unique keys
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)

            if val1 is None:
                interpolated[key] = val2
            elif val2 is None:
                interpolated[key] = val1
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Linear interpolation for numeric values
                interpolated[key] = val1 + (val2 - val1) * ratio
            else:
                # Use config1 value for non-numeric values
                interpolated[key] = val1

        return interpolated

    def export_config_summary(self, config: Dict[str, Any], model_type: str) -> str:
        """
        Export a human-readable summary of a configuration.

        Args:
            config: Configuration to summarize
            model_type: Type of model

        Returns:
            Formatted summary string
        """
        summary = f"""
Model Configuration Summary
===========================

Model Type: {model_type}
Performance Target: {config.get('performance_target', 'Not specified')}

Configuration Parameters:
{json.dumps(config, indent=2, ensure_ascii=False)}

Key Metrics:
- Hidden Dimension: {config.get('hidden_dim', 'N/A')}
- Learning Rate: {config.get('learning_rate', 'N/A')}
"""

        if model_type == "rmm":
            summary += f"- Number of Slots: {config.get('num_slots', 'N/A')}\n"
            summary += f"- Slot Size: {config.get('slot_size', 'N/A')}\n"
            summary += f"- Number of Objects: {config.get('num_objects', 'N/A')}\n"
        elif model_type == "hmm":
            summary += f"- Number of Layers: {config.get('num_layers', 'N/A')}\n"
            summary += f"- Memory Size: {config.get('memory_size', 'N/A')}\n"
            summary += f"- Attention Heads: {config.get('attention_heads', 'N/A')}\n"

        return summary.strip()
