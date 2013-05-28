#!/bin/bash
cd bin 
selfName=`basename $0`                                    #prevent self-run and inifite loop
for fileName in *
 do
  executable=$fileName
   if [[ -x $fileName ]] && [[ $fileName != $selfName  ]]
    then 
    ./$executable
   fi
done
cd ..
