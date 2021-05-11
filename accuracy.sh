#!/bin/sh
# 1 is dir, 2 is how many
total=0
for i in $(seq ${2}); do
	accuracy=$(cat $1/$i/accuracy.txt)
	total=$total+$accuracy
	#cat ${3}/${i}/accuracy.txt >> ${4}
done
echo $total
