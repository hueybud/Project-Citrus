#!/bin/bash

# Get arguments
argument1="$1"
argument2="$2"

# Create directory if it doesn't exist
mkdir -p "$argument2"
cd "$argument2" || exit 1

# Create .cit archive using tar with gzip compression
tar -czf "${argument1}.cit" output.dtm output.dtm.sav output.json

# Verify the archive was created successfully
if [ $? -eq 0 ]; then
    echo "Successfully created ${argument1}.cit"
    # Clean up files only if tar succeeded
    rm -f output.dtm output.dtm.sav output.json
else
    echo "Error: Failed to create archive" >&2
    exit 1
fi
