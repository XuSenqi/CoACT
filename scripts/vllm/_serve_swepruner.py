"""Launcher for the SWEPruner FastAPI service with a transformers 5.x shim.

swe-pruner 0.1.1 skips ``post_init()`` when loading from a pretrained
checkpoint, but transformers 5.x sets ``all_tied_weights_keys`` inside
``post_init`` and then relies on it during ``from_pretrained`` finalization.
Without that attribute the server crashes with::

    AttributeError: 'SwePrunerForCodePruning' object has no attribute
    'all_tied_weights_keys'

SWEPruner has no tied weights (it's a reranker), so we install ``{}`` as a
class-level default before importing the server. Once upstream picks up the
new transformers API this shim can be removed.
"""

from __future__ import annotations

import argparse
import os
import sys

import swe_pruner.swepruner as _swepruner_module
import uvicorn

if not hasattr(_swepruner_module.SwePrunerPreTrainedModel, "all_tied_weights_keys"):
    _swepruner_module.SwePrunerPreTrainedModel.all_tied_weights_keys = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SWEPruner with the transformers 5.x shim")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    os.environ["SWEPRUNER_MODEL_PATH"] = args.model_path

    # Import after the shim and the env var are in place.
    from swe_pruner.online_serving import app

    print(f"Starting SWEPruner on {args.host}:{args.port}", flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
