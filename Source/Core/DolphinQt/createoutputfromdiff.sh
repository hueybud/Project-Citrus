cd "$3" || exit
../hpatchz -f base.sav "$1" "$2" # should this be a single ./ instead of ../ ?
