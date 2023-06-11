
#!/bin/bash
if [ -f "$1" ]; then
    echo "File exists. Deleting $1"
    rm "$1"
fi

echo "Creating new $1"
touch "$1"

while true; do
    nvidia-smi >> "$1"
    echo "&" >> "$1"
    sleep 0.1
done