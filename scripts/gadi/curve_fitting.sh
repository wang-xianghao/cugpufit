SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
EXAMPLEPATH=$SCRIPTPATH/../../examples/curve_fitting.py

if [ -z "$1" ]; then
    ngpu=1
else
    ngpu=$1
fi

echo legate --profile --cpus 16 \
    --gpus $ngpu --sysmem 256000 \
    --fbmem 30000 \
    --eager-alloc-percentage 10 \
    $EXAMPLEPATH

legate --profile --cpus 16 \
    --gpus $ngpu --sysmem 256000 \
    --fbmem 30000 \
    --eager-alloc-percentage 10 \
    $EXAMPLEPATH