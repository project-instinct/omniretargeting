# Adding New Source Adapters

This guide explains how to add support for new motion data formats to OmniRetargeting.

## Overview

OmniRetargeting uses a registry-based architecture where source adapters register themselves and provide motion data through a common interface.

## Source Adapter Interface

All source adapters must inherit from \`DataSource\` and implement the required interface.

## Registration

Register your adapter with the registry using \`register_data_source()\`.

## Example: BVH Adapter

See \`omniretargeting/data_sources/smplx.py\` for a complete example implementation.

## Best Practices

1. Lazy loading for memory efficiency
2. Clear error messages for invalid files
3. Implement source_height for proper scaling
4. Store format-specific data in metadata dict
5. Document expected file format and config options
