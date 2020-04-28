#!/bin/bash

set -e

# Add python as command if needed
if [ "${1:0:1}" = '-' ]; then
    set -- python "$@"
fi

# Drop root privileges if we are running gunicorn
# allow the container to be started with `--user`
if [ "$1" = 'python' -a "$(id -u)" = '0' ]; then
    # Change the ownership of user-mutable directories to gunicorn
    for path in \
        /app \
        /usr/local/cuda/ \
    ; do
        chown -R cuda9:root "$path"
    done
    
    set -- su-exec python "$@"
    #exec su-exec elasticsearch "$BASH_SOURCE" "$@"
fi

# As argument is not related to gunicorn,
# then assume that user wants to run his own process,
# for example a `bash` shell to explore this image
exec "$@"

