
#!/bin/bash
output_file=$1

while true; do
    nvidia-smi >> "$output_file"
    echo "&" >> "$output_file"
    sleep 0.1
done