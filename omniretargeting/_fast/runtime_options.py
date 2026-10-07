"""Optional timing diagnostics; fast always uses the production solver path."""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class RuntimeOptions:
    performance_trace: bool = False

    @classmethod
    def from_environment(cls):
        enabled = set(filter(None, os.environ.get('OMNIRETARGETING_FAST_ENABLE', '').split(',')))
        unknown = enabled - {'performance_trace'}
        if unknown:
            raise ValueError(f'Unknown fast diagnostics: {sorted(unknown)}')
        return cls(performance_trace='performance_trace' in enabled)

    def describe(self):
        return asdict(self)


OPTIONS = RuntimeOptions.from_environment()
