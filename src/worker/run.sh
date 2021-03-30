#!/bin/bash

# need this because of spamming OpenBLAS warnings :(
exec "$@" 2>/dev/null
