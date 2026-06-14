# infinity-provider-pi

A standalone Infinity model provider that delegates model listing/invocation to pi's model registry and API providers.

## Install

```bash
npm install
npm run build
```

## Configure Infinity manually

This provider is not meant to be installed with `infinity provider install --crate`. Add it to `~/.infinity/providers.json` yourself with a custom command:

```json
{
  "pi": {
    "command": ["node", "/absolute/path/to/infinity-provider-pi/dist/main.js"]
  }
}
```

Or, after `npm link` / global install:

```json
{
  "pi": {
    "command": ["infinity-provider-pi"]
  }
}
```

The process prints the socket path as its first stdout line, then serves Infinity provider requests over that socket. Diagnostics go to stderr.

## Model IDs

Models are exposed as `provider/model-id` (for example `anthropic/claude-sonnet-4-5`) so pi providers with overlapping model IDs do not collide.

By default only pi models with configured auth are listed. Set `PI_INFINITY_INCLUDE_UNAVAILABLE=1` in the command environment if you want to list every built-in/custom pi model.
