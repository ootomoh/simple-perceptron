#!/bin/sh

for i in `seq 1 100`
do
	./main | grep elapsed | sed -e 's/.*=\s//g' -e 's/\[.*//g' >> result.csv
done
