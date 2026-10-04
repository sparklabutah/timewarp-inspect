#!/bin/bash
# Start the TimeWarp servers needed by one sample, on the ports used by the
# upstream `scripts/environment/run_all_env.sh` (Wiki 5000, News 5001, Shop 5002).
#
#   TIMEWARP_UI_VERSION  UI version to serve, 1-6.
#   TIMEWARP_SITES       Comma-separated subset of: wiki,news,webshop.
set -eu

version="${TIMEWARP_UI_VERSION:?TIMEWARP_UI_VERSION must be set}"
sites="${TIMEWARP_SITES:?TIMEWARP_SITES must be set}"

case "$version" in
  [1-6]) ;;
  *) echo "TIMEWARP_UI_VERSION must be 1-6, got '$version'" >&2; exit 2 ;;
esac

env_dir=/opt/timewarp/env
started=0
for site in $(echo "$sites" | tr ',' ' '); do
  case "$site" in
    wiki)
      (cd "$env_dir/wiki" && exec python wiki_app.py "-$version" --port=5000) &
      ;;
    news)
      (cd "$env_dir/news" && exec python news_app.py "-$version" --port=5001) &
      ;;
    webshop)
      (cd "$env_dir/webshop" && exec python -m web_agent_site.app "$version" --port=5002 --attrs) &
      ;;
    *)
      echo "Unknown TimeWarp site '$site'" >&2
      exit 2
      ;;
  esac
  started=$((started + 1))
done

if [ "$started" -eq 0 ]; then
  echo "TIMEWARP_SITES is empty" >&2
  exit 2
fi

# Exit (and fail the healthcheck) as soon as any server stops.
wait -n
echo "A TimeWarp server exited" >&2
exit 1
