#!/bin/bash
cd bin 
selfName=`basename $0`                                    #prevent self-run and inifite loop
for fileName in *
 do
   if [[ -x $fileName ]] && [[ $fileName != $selfName  ]]
    then 
    ./$fileName
   fi
done
cd ..
