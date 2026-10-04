# Adding source adapters

A source adapter converts one motion format into the source-neutral contract in
`omniretargeting/data_sources/base.py`. It reports the source as loaded; it does
not optimize, repair, rescale, or rewrite it.

## Contract

Create `omniretargeting/data_sources/<source_type>.py` with a `DataSource`
subclass. At minimum implement `iter_frames()` and expose:

- `target_names`: names in the same order as `MotionFrame.positions`;
- `framerate`: positive frames per second, or `None` only when genuinely unknown;
- `source_height` and `human_height`: source metric height when available;
- `metadata`: source observations such as surfaces, object transforms, inertia,
  contact markers, and provenance.

The inherited `load()` stacks frames into `MotionData`. Positions, root
translations, quaternions (`wxyz`), object points, and metadata arrays must use
one timeline and finite world-frame metric values. Put format-specific keys in
metadata instead of adding source types to generic solver code.

## Factory and registration

```python
from omniretargeting.data_sources.base import DataSource
from omniretargeting.data_sources.registry import register_data_source


def create_my_source(motion_file, source_config, runtime_options):
    return MyDataSource(motion_file=motion_file, **source_config)


register_data_source("my_source", create_my_source, extensions=[".myext"])
```

`source_config` selects format and model facts. `runtime_options` carries caller
controls such as frame ranges. Registration is lazy: requesting source type
`my_source` imports `omniretargeting.data_sources.my_source`.

## Tests

Add focused loader tests beside the existing tests in `tests/data_sources/`.
Cover a small fixture, invalid or incomplete input, timeline consistency,
metadata shape, registration, and—when the format has ambiguous frame
numbering—the chosen provenance semantics. Do not test solver behavior in the
adapter suite.
