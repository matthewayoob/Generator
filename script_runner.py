count=1
for input_file in "$input_folder"/*; do
    if [ -f "$input_file" ]; then
        filename=$(basename -- "$input_file")
        filename_without_extension="${filename%.*}"
        output_file="$output_folder/trained${count}.csv"

        echo "Processing $input_file"
        python k_means.py convert "$input_file" "$output_file"

        ((count++))
    fi
done
