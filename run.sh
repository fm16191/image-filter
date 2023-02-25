#!/bin/bash

# Must be evaled ?

[[ $0 != "$BASH_SOURCE" ]] || { echo -e "Script must be source, please \". $0\" instead."; exit 0; }

free_image_path="../FreeImage"

if [ ! -d $free_image_path ]; then
    echo -e "Must install the FreeImage lib.\nReturn to continue..."
    read
    tar -xf FreeImage3180.tar.gz
    cd FreeImage && make && make install
    export LD_LIBRARY_PATH=$HOME/softs/FreeImage/lib
fi

list=$(module list)
(echo "$list" | grep cuda > /dev/null) || { echo "Loading cuda ..." && module load cuda; }
(echo "$list" | grep "gcc/10.2.0" > /dev/null) || { echo "Loading gcc@10.2.0 ..." && module load gcc/10.2.0; }

make && ./modif_img.exe
