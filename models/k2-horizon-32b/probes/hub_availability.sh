#!/usr/bin/env bash
# What IFM has actually published on the Hugging Face Hub, as opposed to what
# the model card says is published or "will be made public". Unauthenticated,
# so it reports what any reader can reach. A 401 from the Hub API means the
# repository is private, gated behind login, or does not exist; either way it
# is not public.
#
#   bash models/k2-horizon-32b/probes/hub_availability.sh | tee models/k2-horizon-32b/results/hub-availability.log
set -u
api=https://huggingface.co/api
date -u '+checked %Y-%m-%dT%H:%MZ'
echo "== datasets named in the K2-Horizon-32B card"
for d in IFM/K2-Horizon-Pretrain-Data IFM/K2-Horizon-Midtrain-Data; do
  printf '  %-32s HTTP %s\n' "$d" "$(curl -s -o /dev/null -w '%{http_code}' "$api/datasets/$d")"
done
echo "== branches of IFM/K2-Horizon-32B (the card's table lists intermediate checkpoints as branches)"
curl -s "$api/models/IFM/K2-Horizon-32B/refs" | python3 -c 'import json,sys; print("  ", [b["name"] for b in json.load(sys.stdin)["branches"]])'
echo "== IFM's public datasets"
curl -s "$api/datasets?author=IFM&limit=100" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ", len(d), "public:", ", ".join(x["id"].split("/")[1] for x in d))'
echo "== K2-Horizon models IFM has published"
curl -s "$api/models?author=IFM&search=K2-Horizon&limit=100" | python3 -c 'import json,sys; print("  ", ", ".join(sorted(x["id"].split("/")[1] for x in json.load(sys.stdin))))'
