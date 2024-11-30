#!/bin/bash

# Used in server machine(s)

files=(
"test_data_X.pkl" "test_data_y.pkl" "agg_gradients.pkl"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "File '$file' removed"
    else
        echo "File '$file' not found"
    fi
done

for ((i=1; i<=4; i++)); do
    c_files=(
        "client_acc_${i}.pkl" "client_dataset_${i}.pkl" "local_gradients_${i}.pkl"
    )
    for f in "${c_files[@]}"; do
        if [ -f "$f" ]; then
            rm "$f"
            echo "File '$f' removed"
        else
            echo "File '$f' not found"
        fi
    done
done

cd LA/

# if the last command was successful, clear the folder
if [ $? -eq 0 ]; then
    rm *
    echo "Folder LA/ cleared"
else
    echo "Error in clearing folder LA/"
fi