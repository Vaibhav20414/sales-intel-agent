"""Sales Intelligence Agent: multi-step, tool-using sales analysis with a live trace."""
from .agent_offline import run_offline_agent
from .config import AgentConfig
from .data import DataError, load_sales
from .tools import TOOL_REGISTRY, SalesContext
from .trace import TraceEvent, Tracer

__all__ = [
    "AgentConfig",
    "DataError",
    "SalesContext",
    "TOOL_REGISTRY",
    "TraceEvent",
    "Tracer",
    "load_sales",
    "run_offline_agent",
]
