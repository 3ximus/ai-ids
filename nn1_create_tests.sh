#!/bin/bash
set -e
# args[1:-1]: pcap files, arg[-1]: n_features
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
for file in "${@:1:${#}-1}"; do
	cd dist/bin
	./CICFlowMeter $file ../../csv/test/extracted/
	cd ../..
done

[[ -d "csv/test/compacted/${@:$#}features/" ]] || mkdir csv/test/compacted/${@:$#}features/
if [ "$3" == "BENIGN" ] ; then
	python scripts/compact_flows.py csv/test/extracted/*.csv csv/test/compacted/${@:$#}features/ -f scripts/features/${@:$#}.txt --benign
else
	python scripts/compact_flows.py csv/test/extracted/*.csv csv/test/compacted/${@:$#}features/ -f scripts/features/${@:$#}.txt
fi