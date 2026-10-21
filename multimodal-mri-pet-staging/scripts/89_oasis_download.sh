#!/bin/bash
# Download the OASIS-1 cross-sectional archives (open access) on a login node. One instance only (lock file on the shared
# file system); every archive is downloaded to a temporary name, checked against the server's size and with gzip -t.
D=$HOME/alz/oasis1/dl; mkdir -p $D; cd $D
exec 9> $D/.lock; flock -n 9 || { echo "another download is running"; exit 0; }
U=https://download.nrg.wustl.edu/data
for i in $(seq 1 12); do
  f=oasis_cross-sectional_disc$i.tar.gz
  [ -f $f.ok ] && continue
  want=$(curl -sI -m 30 $U/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  rm -f $f.part
  curl -s -m 3000 -o $f.part $U/$f
  got=$(stat -c %s $f.part 2>/dev/null)
  if [ -n "$want" ] && [ "$got" = "$want" ] && gzip -t $f.part 2>/dev/null; then mv $f.part $f; touch $f.ok; echo "$(date +%T) $f ok $got"; else echo "$(date +%T) $f BAD got=$got want=$want"; fi
done
echo "ALLDONE $(ls *.ok 2>/dev/null | wc -l)/12"
