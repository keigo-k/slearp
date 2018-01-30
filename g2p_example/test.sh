#!/bin/csh

set C = 100

foreach b ( 0.0075 0.01 0.0125 )
../slearp -t train.align -d dev.align -cn 5 -jn 2 -w g2p.model.$C.$b -m sscw -C $C -base $b > & log.sscw.$C.$b.d &
end

