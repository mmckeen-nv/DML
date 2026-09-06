"""Host interfaces shared by standalone Hermes plugin unit tests."""
import sys
import types
from pathlib import Path


# The provider is shipped into Hermes, but this repository's CI intentionally
# tests its deterministic policy without installing Hermes itself. Supply the
# minimum host interfaces needed to import the plugin module.
agent_module = types.ModuleType("agent")
memory_provider_module = types.ModuleType("agent.memory_provider")
setattr(memory_provider_module, "MemoryProvider", object)
setattr(agent_module, "memory_provider", memory_provider_module)
sys.modules.setdefault("agent", agent_module)
sys.modules.setdefault("agent.memory_provider", memory_provider_module)

constants_module = types.ModuleType("hermes_constants")
setattr(constants_module, "get_default_hermes_root", lambda: Path.home() / ".hermes")
setattr(constants_module, "get_hermes_home", lambda: Path.home() / ".hermes")
sys.modules.setdefault("hermes_constants", constants_module)

hermes_cli_module = types.ModuleType("hermes_cli")
config_module = types.ModuleType("hermes_cli.config")
setattr(config_module, "cfg_get", lambda *_args, **_kwargs: None)
setattr(config_module, "load_config", lambda: {})
setattr(hermes_cli_module, "config", config_module)
sys.modules.setdefault("hermes_cli", hermes_cli_module)
sys.modules.setdefault("hermes_cli.config", config_module)
